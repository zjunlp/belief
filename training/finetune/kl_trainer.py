import copy
import json
import random
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer


class TokenizedQASFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_seq_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError("Dataset file must be a JSON list")

        self._items: List[Dict[str, Any]] = []
        for ex in raw:
            if not isinstance(ex, dict):
                continue
            problem = ex.get("problem") or ex.get("question") or ex.get("original_problem")
            answer = ex.get("answer") or ex.get("golden_answer") or ex.get("answers") or ex.get("original_answer")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            if problem is None or answer is None:
                continue

            prompt_messages = [{"role": "user", "content": str(problem)}]
            full_messages = [
                {"role": "user", "content": str(problem)},
                {"role": "assistant", "content": str(answer)},
            ]

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )

            if len(full_ids) > self.max_seq_length:
                full_ids = full_ids[: self.max_seq_length]

            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            self._items.append(
                {
                    "input_ids": list(full_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._items[idx]


class TwoDatasetWrapper(Dataset):
    def __init__(self, sft_dataset: Dataset, kl_dataset: Dataset, kl_sampling: str = "random"):
        self.sft_dataset = sft_dataset
        self.kl_dataset = kl_dataset
        self._len = max(len(self.sft_dataset), len(self.kl_dataset))
        self.sft_len = len(self.sft_dataset)
        self.kl_len = len(self.kl_dataset)
        self.kl_sampling = "paired" if kl_sampling == "sequential" else kl_sampling

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sft_idx = idx % self.sft_len
        sft_item = self.sft_dataset[sft_idx]

        if self.kl_sampling == "paired":
            kl_idx = idx % self.kl_len
        elif self.kl_sampling == "random":
            kl_idx = random.randrange(self.kl_len)
        else:
            raise ValueError(f"Unknown kl_sampling: {self.kl_sampling}. Expected 'random', 'paired', or 'sequential'.")

        kl_item = self.kl_dataset[kl_idx]
        return sft_item, kl_item


def _pad_2d(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


class TwoDatasetDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        sft_features = [f[0] for f in features]
        kl_features = [f[1] for f in features]

        sft_input_ids = [x["input_ids"] for x in sft_features]
        sft_labels = [x["labels"] for x in sft_features]
        kl_input_ids = [x["input_ids"] for x in kl_features]
        kl_labels = [x["labels"] for x in kl_features]

        sft_input_ids_t = _pad_2d(sft_input_ids, self.tokenizer.pad_token_id)
        sft_labels_t = _pad_2d(sft_labels, -100)
        kl_input_ids_t = _pad_2d(kl_input_ids, self.tokenizer.pad_token_id)
        kl_labels_t = _pad_2d(kl_labels, -100)

        sft_attn = (sft_input_ids_t != self.tokenizer.pad_token_id).long()
        kl_attn = (kl_input_ids_t != self.tokenizer.pad_token_id).long()

        return {
            "sft": {
                "input_ids": sft_input_ids_t,
                "attention_mask": sft_attn,
                "labels": sft_labels_t,
            },
            "kl": {
                "input_ids": kl_input_ids_t,
                "attention_mask": kl_attn,
                "labels": kl_labels_t,
            },
        }


class KLDivergenceTrainer(Trainer):
    def __init__(
        self,
        *args,
        kl_weight: float = 0.1,
        kl_temperature: float = 1.0,
        ref_model: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        self.ref_model = ref_model
        super().__init__(*args, **kwargs)

        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

            if getattr(self.args, "deepspeed", None):
                self.ref_model = self._prepare_ref_model_deepspeed(self.ref_model)

    def _prepare_ref_model_deepspeed(self, model: torch.nn.Module) -> torch.nn.Module:
        try:
            import deepspeed  # type: ignore
        except Exception:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model

        if not hasattr(self, "accelerator"):
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is None:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
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
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _masked_kl(self, student_logits: torch.Tensor, ref_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        student_logits = student_logits[:, :-1, :]
        ref_logits = ref_logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        mask = shift_labels.ne(-100)
        if mask.sum().item() == 0:
            return student_logits.new_zeros(())

        t = float(self.kl_temperature)
        log_q = F.log_softmax(student_logits / t, dim=-1)
        log_p = F.log_softmax(ref_logits / t, dim=-1)

        per_vocab = F.kl_div(log_q, log_p, log_target=True, reduction="none")
        per_token = per_vocab.sum(dim=-1)

        per_token = per_token * mask.to(per_token.dtype)
        loss = per_token.sum() / mask.sum().to(per_token.dtype)
        return loss * (t * t)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sft_inputs = inputs["sft"]
        kl_inputs = inputs["kl"]

        outputs_sft = model(
            input_ids=sft_inputs["input_ids"],
            attention_mask=sft_inputs.get("attention_mask"),
            labels=sft_inputs.get("labels"),
        )
        loss_sft = outputs_sft.loss

        student_outputs = model(
            input_ids=kl_inputs["input_ids"],
            attention_mask=kl_inputs.get("attention_mask"),
        )

        with torch.no_grad():
            if self.ref_model is not None:
                ref_outputs = self.ref_model(
                    input_ids=kl_inputs["input_ids"],
                    attention_mask=kl_inputs.get("attention_mask"),
                )
            else:
                # FIXME: IF THIS CONDITION IS MET, WE NEED A REFERENCE MODEL for proper KL divergence computation
                raise ValueError("Reference model is not available for KL divergence computation")
                unwrapped = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
                ctx = unwrapped.disable_adapter() if hasattr(unwrapped, "disable_adapter") else nullcontext()
                with ctx:
                    ref_outputs = model(
                        input_ids=kl_inputs["input_ids"],
                        attention_mask=kl_inputs.get("attention_mask"),
                    )

        loss_kl = self._masked_kl(student_outputs.logits, ref_outputs.logits, kl_inputs.get("labels"))
        loss = loss_sft + (float(self.kl_weight) * loss_kl)

        return (loss, outputs_sft) if return_outputs else loss
