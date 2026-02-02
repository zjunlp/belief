import torch

from context_distill_kl_trainer import ContextDistillKLTrainer

class ContextDistillKLSFTTrainer(ContextDistillKLTrainer):
    def __init__(
        self,
        *args,
        sft_weight: float = 1.0,
        **kwargs,
    ):
        self.sft_weight = sft_weight
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sft_inputs = inputs.get("sft")
        if sft_inputs is None:
            raise ValueError("Expected inputs to contain 'sft' key for SFT loss.")

        outputs_sft = model(
            input_ids=sft_inputs["input_ids"],
            attention_mask=sft_inputs.get("attention_mask"),
            labels=sft_inputs.get("labels"),
        )
        loss_sft = outputs_sft.loss

        base_loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
        if (self.state.global_step+1) % self.args.logging_steps == 0:
            self.log({
            "sft_loss": loss_sft.detach().item(),
        })
        loss = base_loss + (float(self.sft_weight) * loss_sft)

        if return_outputs:
            return loss, outputs_sft
        return loss

class ContextDistillKLKLTrainer(ContextDistillKLTrainer):
    def __init__(
        self,
        *args,
        sft_weight: float = 1.0,
        **kwargs,
    ):
        self.sft_weight = sft_weight
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sft_inputs = inputs.get("sft")
        if sft_inputs is None:
            raise ValueError("Expected inputs to contain 'sft' key for SFT loss.")
        # base_loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
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
                ref_outputs = self.teacher_model(input_ids=kl_input_ids, attention_mask=kl_attention_mask)
                ref_logits = ref_outputs.logits

            loss_kl = self._masked_kl_from_labels(kl_student_logits, ref_logits, kl_labels.to(kl_input_ids.device), self.kl_temperature)

        loss_sft = student_logits.new_zeros(())
        if float(self.sft_weight) != 0.0:
            sft_input_ids = sft_inputs["input_ids"]
            sft_attention_mask = sft_inputs.get("attention_mask")
            sft_labels = sft_inputs.get("labels")
            if sft_labels is None:
                raise ValueError("SFT batch must include 'labels' for masking.")

            sft_student_outputs = model(input_ids=sft_input_ids, attention_mask=sft_attention_mask)
            sft_student_logits = sft_student_outputs.logits

            with torch.no_grad():
                if self.ref_model is not None:
                    ref_outputs = self.ref_model(input_ids=sft_input_ids, attention_mask=sft_attention_mask)
                else:
                    ref_outputs = self.teacher_model(input_ids=sft_input_ids, attention_mask=sft_attention_mask)
                ref_logits = ref_outputs.logits

            loss_sft = self._masked_kl_from_labels(
                sft_student_logits,
                ref_logits,
                sft_labels.to(sft_input_ids.device),
                self.kl_temperature,
            )
        if (self.state.global_step+1) % self.args.logging_steps == 0:
            self.log({
                "distill_loss": loss_distill.detach().item(),
                "qakl_loss": loss_kl.detach().item(),
                "wikikl_loss": loss_sft.detach().item(),
            })
        loss = loss_distill + (float(self.kl_weight) * loss_kl) + (float(self.sft_weight) * loss_sft)

        if return_outputs:
            return loss, student_outputs
        return loss
