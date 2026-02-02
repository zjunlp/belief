#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   context_distillation_trainer.py
@Time    :   2025/10/13
@Author  :   haoming
@Version :   1.0
@Desc    :   Custom trainer integrating CONTEXT-INJECTION method for context injection
'''
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from typing import Dict, Optional, Any

class ContextDistillationTrainer(SFTTrainer):
    """
    Custom Trainer that implements CONTEXT-DISTRillation knowledge distillation.
    
    This trainer uses a dual-model approach:
    - Base model (teacher): Has access to context, outputs serve as targets
    - Fine-tuned model (student): No context access, learns to mimic teacher
    
    Args:
        base_model: The frozen base model (teacher) that has access to context
        distill_temperature: Temperature for knowledge distillation (default: 1.0)
        use_distillation: Whether to use distillation loss (default: True)
        **kwargs: Additional arguments passed to Trainer
    """
    
    def __init__(
        self,
        distill_temperature: float = 1.0,
        use_distillation: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.distill_temperature = distill_temperature
        self.use_distillation = use_distillation
    
    def _prepare_dataset(self, dataset, tokenizer, max_seq_length, preprocessing_num_workers, overwrite_cache, model_max_length, is_eval=False):
        # For CONTEXT-INJECTION, we handle truncation in our dataset preprocessing
        # Just return the dataset as-is
        return dataset
    
    def get_train_dataloader(self):
        """Override to ensure our custom data collator is used."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # Use our custom data collator
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def distillation_loss(self, student_logits, teacher_logits, temperature=None):
        """
        Compute KL divergence loss for knowledge distillation.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model  
            temperature: Temperature for softening distributions
            
        Returns:
            Distillation loss value
        """
        if temperature is None:
            temperature = self.distill_temperature
        
        # KL divergence between teacher and student distributions
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),  # log Q
            F.log_softmax(teacher_logits / temperature, dim=-1),  # log P
            log_target=True,  # Tell function: target is log P
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with optional knowledge distillation.

        If use_distillation is True and context is provided in inputs:
            - Teacher model sees: context + question + answer
            - Student model sees: question + answer
            - Loss is KL divergence between their answer distributions
        Otherwise:
            - Standard causal language modeling loss
        """
        has_context = "teacher_input_ids" in inputs and "student_input_ids" in inputs
        if self.use_distillation and has_context:
            teacher_input_ids = inputs.get("teacher_input_ids")  # Full input with context
            student_input_ids = inputs.get("student_input_ids")  # Input without context
            teacher_attention_mask = inputs.get("teacher_attention_mask")
            student_attention_mask = inputs.get("student_attention_mask")
            answer_length = inputs.get("answer_length")
            teacher_question_length = inputs.get("teacher_question_length")
            student_question_length = inputs.get("student_question_length")

            unwrapped_model = self.accelerator.unwrap_model(model)
            model.train()
            with torch.no_grad():
                # Use disable_adapter() to use base model for teacher (without LoRA)
                with unwrapped_model.disable_adapter():
                    teacher_outputs = model(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask
                    )
                    teacher_logits = teacher_outputs.logits
            
            # === 3. Student Forward (Fine-tuned Model) ===
            # Student model uses the active LoRA adapter by default
            student_outputs = model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask
            )
            student_logits = student_outputs.logits
            
            if answer_length is not None:
                total_loss = 0
                batch_size = student_logits.size(0)
                for i in range(batch_size):
                    ans_len = int(answer_length[i].item()) if answer_length.numel() > 1 else int(answer_length.item())
                    s_qlen = int(student_question_length[i].item()) if student_question_length.numel() > 1 else int(student_question_length.item())
                    t_qlen = int(teacher_question_length[i].item()) if teacher_question_length.numel() > 1 else int(teacher_question_length.item())
                    
                    s_start = s_qlen
                    t_start = t_qlen
                    
                    s_end = min(s_start + ans_len, student_logits.size(1))
                    t_end = min(t_start + ans_len, teacher_logits.size(1))
                    
                    common_len = min(s_end - s_start, t_end - t_start)
                    
                    s_logit = student_logits[i, s_start : s_start + common_len, :]
                    t_logit = teacher_logits[i, t_start : t_start + common_len, :]
                    
                    # Compute loss for each item separately to avoid padding effects
                    total_loss += self.distillation_loss(s_logit.unsqueeze(0), t_logit.unsqueeze(0))
                
                loss = total_loss / batch_size
                
            else:
                seq_len = min(student_logits.size(1), teacher_logits.size(1))
                aligned_student_logits = student_logits[:, -seq_len:, :]
                aligned_teacher_logits = teacher_logits[:, -seq_len:, :]
                loss = self.distillation_loss(aligned_student_logits, aligned_teacher_logits)
            
            return (loss, student_outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
