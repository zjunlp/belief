import copy
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer


class ContextDistillKLTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model: Optional[torch.nn.Module] = None,
        ref_model: Optional[torch.nn.Module] = None,
        distill_temperature: float = 1.0,
        kl_weight: float = 0.1,
        kl_temperature: float = 1.0,
        **kwargs,
    ):
        self.teacher_model = teacher_model
        self.ref_model = ref_model
        self.distill_temperature = distill_temperature
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        super().__init__(*args, **kwargs)

        if self.teacher_model is None:
            raise ValueError("teacher_model is required")

        self._freeze_eval(self.teacher_model)
        if self.ref_model is not None:
            self._freeze_eval(self.ref_model)

        if getattr(self.args, "deepspeed", None):
            self.teacher_model = self._prepare_aux_model_deepspeed(self.teacher_model)
            if self.ref_model is not None:
                self.ref_model = self._prepare_aux_model_deepspeed(self.ref_model)

    def _freeze_eval(self, model: torch.nn.Module) -> None:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    def _prepare_aux_model_deepspeed(self, model: torch.nn.Module) -> torch.nn.Module:
        try:
            import deepspeed  # type: ignore
        except Exception:
            self._freeze_eval(model)
            return model

        if not hasattr(self, "accelerator"):
            self._freeze_eval(model)
            return model

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is None:
            self._freeze_eval(model)
            return model

        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None and hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs.get("zero_optimization", {}).get("stage", 0) == 3:
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

        if config_kwargs.get("zero_optimization", {}).get("stage", 0) != 3:
            config_kwargs.setdefault("zero_optimization", {})
            config_kwargs["zero_optimization"]["stage"] = 0

        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        self._freeze_eval(model)
        return model

    def _gather_answer_logits(
        self,
        logits: torch.Tensor,
        q_lens: torch.Tensor,
        ans_lens: torch.Tensor,
    ):
        bsz, seq_len, vocab = logits.shape
        max_ans = int(ans_lens.max().item()) if ans_lens.numel() > 0 else 0
        if max_ans <= 0:
            empty = logits.new_zeros((bsz, 0, vocab))
            mask = logits.new_zeros((bsz, 0), dtype=torch.bool)
            return empty, mask

        start = torch.clamp(q_lens - 1, min=0)
        offsets = torch.arange(max_ans, device=logits.device).unsqueeze(0).expand(bsz, -1)
        idx = start.unsqueeze(1) + offsets

        valid = (offsets < ans_lens.unsqueeze(1)) & (idx >= 0) & (idx < seq_len)
        idx = idx.clamp(min=0, max=seq_len - 1)

        gathered = logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, vocab))
        return gathered, valid

    def _masked_kl(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
        if mask.numel() == 0 or mask.sum().item() == 0:
            return student_logits.new_zeros(())

        t = float(temperature)
        log_q = F.log_softmax(student_logits / t, dim=-1)
        log_p = F.log_softmax(teacher_logits / t, dim=-1)

        per_vocab = F.kl_div(log_q, log_p, log_target=True, reduction="none")
        per_token = per_vocab.sum(dim=-1)

        per_token = per_token * mask.to(per_token.dtype)
        loss = per_token.sum() / mask.sum().to(per_token.dtype)
        return loss * (t * t)

    def _masked_kl_from_labels(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        student_logits = student_logits[:, :-1, :]
        teacher_logits = teacher_logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        mask = shift_labels.ne(-100)
        if mask.sum().item() == 0:
            return student_logits.new_zeros(())

        t = float(temperature)
        log_q = F.log_softmax(student_logits / t, dim=-1)
        log_p = F.log_softmax(teacher_logits / t, dim=-1)

        per_vocab = F.kl_div(log_q, log_p, log_target=True, reduction="none")
        per_token = per_vocab.sum(dim=-1)

        per_token = per_token * mask.to(per_token.dtype)
        loss = per_token.sum() / mask.sum().to(per_token.dtype)
        return loss * (t * t)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cd_inputs = inputs.get("context_distillation")
        kl_inputs = inputs.get("kl")

        if cd_inputs is None or kl_inputs is None:
            raise ValueError("Expected inputs to contain 'context_distillation' and 'kl' keys.")

        student_input_ids = cd_inputs["student_input_ids"]
        student_attention_mask = cd_inputs.get("student_attention_mask")
        teacher_input_ids = cd_inputs["teacher_input_ids"]
        teacher_attention_mask = cd_inputs.get("teacher_attention_mask")
        student_q_len = cd_inputs["student_question_length"].to(student_input_ids.device)
        teacher_q_len = cd_inputs["teacher_question_length"].to(student_input_ids.device)
        answer_len = cd_inputs["answer_length"].to(student_input_ids.device)

        student_outputs = model(input_ids=student_input_ids, attention_mask=student_attention_mask)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask)
            teacher_logits = teacher_outputs.logits

        s_ans_logits, s_mask = self._gather_answer_logits(student_logits, student_q_len, answer_len)
        t_ans_logits, t_mask = self._gather_answer_logits(teacher_logits, teacher_q_len, answer_len)
        mask = s_mask & t_mask
        loss_distill = self._masked_kl(s_ans_logits, t_ans_logits, mask, self.distill_temperature)

        loss_kl = student_logits.new_zeros(())
        if float(self.kl_weight) != 0.0:
            kl_input_ids = kl_inputs["input_ids"]
            kl_attention_mask = kl_inputs.get("attention_mask")
            kl_labels = kl_inputs.get("labels")
            if kl_labels is None:
                raise ValueError("KL batch must include 'labels' for masking.")

            kl_student_outputs = model(input_ids=kl_input_ids, attention_mask=kl_attention_mask)
            kl_student_logits = kl_student_outputs.logits

            with torch.no_grad():
                if self.ref_model is not None:
                    ref_outputs = self.ref_model(input_ids=kl_input_ids, attention_mask=kl_attention_mask)
                else:
                    ref_outputs = self.teacher_model(input_ids=kl_input_ids, attention_mask=kl_attention_mask)
                ref_logits = ref_outputs.logits

            loss_kl = self._masked_kl_from_labels(kl_student_logits, ref_logits, kl_labels.to(kl_input_ids.device), self.kl_temperature)
        if (self.state.global_step+1) % self.args.logging_steps == 0:
            self.log({
            "distill_loss": loss_distill.detach().item(),
            "kl_loss": loss_kl.detach().item(),
        })
        loss = loss_distill + (float(self.kl_weight) * loss_kl)
        return (loss, student_outputs) if return_outputs else loss
    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     cd_inputs = inputs.get("context_distillation")
    #     kl_inputs = inputs.get("kl")

    #     if cd_inputs is None or kl_inputs is None:
    #         raise ValueError("Expected inputs to contain 'context_distillation' and 'kl' keys.")

    #     student_input_ids = cd_inputs["student_input_ids"]
    #     student_attention_mask = cd_inputs.get("student_attention_mask")
    #     teacher_input_ids = cd_inputs["teacher_input_ids"]
    #     teacher_attention_mask = cd_inputs.get("teacher_attention_mask")
    #     student_q_len = cd_inputs["student_question_length"].to(student_input_ids.device)
    #     teacher_q_len = cd_inputs["teacher_question_length"].to(student_input_ids.device)
    #     answer_len = cd_inputs["answer_length"].to(student_input_ids.device)

    #     student_outputs = model(input_ids=student_input_ids, attention_mask=student_attention_mask)
    #     student_logits = student_outputs.logits

    #     with torch.no_grad():
    #         teacher_outputs = self.teacher_model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask)
    #         teacher_logits = teacher_outputs.logits

    #     s_ans_logits, s_mask = self._gather_answer_logits(student_logits, student_q_len, answer_len)
    #     t_ans_logits, t_mask = self._gather_answer_logits(teacher_logits, teacher_q_len, answer_len)
    #     mask = s_mask & t_mask
    #     loss_distill = self._masked_kl(s_ans_logits, t_ans_logits, mask, self.distill_temperature)

    #     loss_kl = student_logits.new_zeros(())
    #     if float(self.kl_weight) != 0.0:
    #         kl_input_ids = kl_inputs["input_ids"]
    #         kl_attention_mask = kl_inputs.get("attention_mask")
    #         kl_labels = kl_inputs.get("labels")
    #         if kl_labels is None:
    #             raise ValueError("KL batch must include 'labels' for masking.")

    #         kl_student_outputs = model(input_ids=kl_input_ids, attention_mask=kl_attention_mask)
    #         kl_student_logits = kl_student_outputs.logits

    #         with torch.no_grad():
    #             if self.ref_model is not None:
    #                 ref_outputs = self.ref_model(input_ids=kl_input_ids, attention_mask=kl_attention_mask)
    #             else:
    #                 ref_outputs = self.teacher_model(input_ids=kl_input_ids, attention_mask=kl_attention_mask)
    #             ref_logits = ref_outputs.logits

    #         loss_kl = self._masked_kl_from_labels(kl_student_logits, ref_logits, kl_labels.to(kl_input_ids.device), self.kl_temperature)

    #     loss = loss_distill + (float(self.kl_weight) * loss_kl)
    #     return (loss, student_outputs) if return_outputs else loss
