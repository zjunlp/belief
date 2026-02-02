#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   mixed_context_kl_trainer.py
@Time    :   2025/01/15
@Author  :   haoming
@Version :   1.0
@Desc    :   Trainer combining context_distillation and KL divergence training
'''
import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, Any
from contextlib import nullcontext
import copy

class MixedContextKLTrainer(Trainer):
    """
    Trainer combining context_distillation and KL divergence.
    
    Trains two types of loss simultaneously:
    1. Context Distillation Loss: teacher has context, student does not
    2. KL Divergence Loss: teacher and student use same input, but teacher is base model
    
    Args:
        distill_temperature: Temperature parameter for context distillation
        kl_weight: Weight for KL divergence loss
        kl_temperature: Temperature parameter for KL divergence
        ref_model: Reference model for KL divergence (base model)
        **kwargs: Other Trainer parameters
    """
    
    def __init__(
        self,
        distill_temperature: float = 1.0,
        kl_weight: float = 0.1,
        kl_temperature: float = 1.0,
        ref_model: Optional[torch.nn.Module] = None,
        **kwargs
    ):
        self.distill_temperature = distill_temperature
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        self.ref_model = ref_model
        super().__init__(**kwargs)
        
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
            
            if getattr(self.args, "deepspeed", None):
                self.ref_model = self._prepare_ref_model_deepspeed(self.ref_model)
    
    def _prepare_ref_model_deepspeed(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare reference model for deepspeed environment"""
        try:
            import deepspeed
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
    
    def context_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Calculate KL divergence loss for context distillation.
        
        Args:
            student_logits: Student model logits (without context)
            teacher_logits: Teacher model logits (with context)
            temperature: Temperature parameter
            
        Returns:
            KL divergence loss
        """
        if temperature is None:
            temperature = self.distill_temperature
        
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.log_softmax(teacher_logits / temperature, dim=-1),
            log_target=True,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return loss
    
    def kl_divergence_loss(self, student_logits: torch.Tensor, ref_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence loss (with mask).
        
        Args:
            student_logits: Student model logits
            ref_logits: Reference model logits
            labels: Labels, -100 indicates positions to ignore
            
        Returns:
            KL divergence loss
        """
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
        """
        Calculate mixed loss: context_distillation loss + KL divergence loss.
        """
        cd_inputs = inputs["context_distillation"]
        kl_inputs = inputs["kl"]
        
        unwrapped_model = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
        
        # ========== 1. Context Distillation Loss ==========
        teacher_input_ids = cd_inputs.get("teacher_input_ids")
        student_input_ids = cd_inputs.get("student_input_ids")
        teacher_attention_mask = cd_inputs.get("teacher_attention_mask")
        student_attention_mask = cd_inputs.get("student_attention_mask")
        answer_length = cd_inputs.get("answer_length")
        teacher_question_length = cd_inputs.get("teacher_question_length")
        student_question_length = cd_inputs.get("student_question_length")
        
        model.train()
        with torch.no_grad():
            # Teacher uses base model (disable adapter)
            with unwrapped_model.disable_adapter():
                teacher_outputs = model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask
                )
                teacher_logits = teacher_outputs.logits
        
        # Student uses fine-tuned model (active adapter)
        student_outputs_cd = model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask
        )
        student_logits_cd = student_outputs_cd.logits
        
        # Calculate context distillation loss (only for answer part)
        if answer_length is not None:
            cd_loss_total = 0
            batch_size = student_logits_cd.size(0)
            for i in range(batch_size):
                ans_len = int(answer_length[i].item()) if answer_length.numel() > 1 else int(answer_length.item())
                s_qlen = int(student_question_length[i].item()) if student_question_length.numel() > 1 else int(student_question_length.item())
                t_qlen = int(teacher_question_length[i].item()) if teacher_question_length.numel() > 1 else int(teacher_question_length.item())
                
                s_start = s_qlen
                t_start = t_qlen
                
                s_end = min(s_start + ans_len, student_logits_cd.size(1))
                t_end = min(t_start + ans_len, teacher_logits.size(1))
                
                common_len = min(s_end - s_start, t_end - t_start)
                
                s_logit = student_logits_cd[i, s_start : s_start + common_len, :]
                t_logit = teacher_logits[i, t_start : t_start + common_len, :]
                
                cd_loss_total += self.context_distillation_loss(s_logit.unsqueeze(0), t_logit.unsqueeze(0))
            
            cd_loss = cd_loss_total / batch_size
        else:
            # Fallback: align at sequence end
            seq_len = min(student_logits_cd.size(1), teacher_logits.size(1))
            aligned_student_logits = student_logits_cd[:, -seq_len:, :]
            aligned_teacher_logits = teacher_logits[:, -seq_len:, :]
            cd_loss = self.context_distillation_loss(aligned_student_logits, aligned_teacher_logits)
        
        # ========== 2. KL Divergence Loss ==========
        kl_input_ids = kl_inputs["input_ids"]
        kl_attention_mask = kl_inputs.get("attention_mask")
        kl_labels = kl_inputs.get("labels")
        
        # Student forward (using fine-tuned model)
        student_outputs_kl = model(
            input_ids=kl_input_ids,
            attention_mask=kl_attention_mask
        )
        student_logits_kl = student_outputs_kl.logits
        
        # Reference model forward
        with torch.no_grad():
            if self.ref_model is not None:
                ref_outputs = self.ref_model(
                    input_ids=kl_input_ids,
                    attention_mask=kl_attention_mask
                )
            else:
                # If ref_model not provided, use disable_adapter
                with unwrapped_model.disable_adapter():
                    ref_outputs = model(
                        input_ids=kl_input_ids,
                        attention_mask=kl_attention_mask
                    )
            ref_logits = ref_outputs.logits
        
        # Calculate KL divergence loss
        kl_loss = self.kl_divergence_loss(student_logits_kl, ref_logits, kl_labels)
        
        # ========== 3. Combined Loss ==========
        total_loss = cd_loss + self.kl_weight * kl_loss
        
        if return_outputs:
            # Return student_outputs for logging
            return (total_loss, student_outputs_cd)
        else:
            return total_loss
