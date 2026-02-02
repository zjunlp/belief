#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   context_injection_trainer.py
@Time    :   2025/10/13
@Author  :   haoming
@Version :   1.0
@Desc    :   Custom trainer integrating CONTEXT-INJECTION method for context injection
'''
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from typing import Dict, Optional, Any
import copy
import torch.nn.utils.rnn as rnn_utils

class ContextInjectionTrainer(SFTTrainer):
    """
    Custom Trainer that implements CONTEXT-INJECTION knowledge distillation.
    
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
        """Override to skip TRL's dataset truncation for CONTEXT-INJECTION."""
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
            F.log_softmax(student_logits / temperature, dim=-1),
            F.log_softmax(teacher_logits / temperature, dim=-1),
            log_target=True,
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
        has_context = "context_input_ids" in inputs
        if self.use_distillation and has_context:
            context_input_ids = inputs.pop("context_input_ids")  # Full input with context
            target_input_ids = inputs["input_ids"]  # Input without context
            context_length = inputs.pop("context_length", None)
            answer_length = inputs.pop("answer_length", None)
            question_length = inputs.pop("question_length", None)
            context_attention_mask = inputs.get("context_attention_mask")
            unwrapped_model = self.accelerator.unwrap_model(model)

            model.train()
            with torch.no_grad():
            # Key: Use disable_adapter context manager
            # Automatically disable LoRA on entry, restore on exit, no need to manually manage requires_grad
                with unwrapped_model.disable_adapter():
                    teacher_outputs = model(
                        input_ids=context_input_ids,
                        attention_mask=context_attention_mask
                    )
                    teacher_logits = teacher_outputs.logits

            # === 3. Student Forward (Fine-tuned Model) ===
            student_outputs = model(**inputs)
            student_logits = student_outputs.logits
            # Align logits for distillation
            aligned_student_logits = []
            aligned_teacher_logits = []
            batch_size = student_logits.size(0)
            student_seq_len = student_logits.size(1)
            teacher_seq_len = teacher_logits.size(1)
            if context_length is not None and answer_length is not None:
                # Only compute distillation loss for answer tokens
                for i in range(batch_size):
                    ctx_len = context_length[i].item()
                    qs_len = question_length[i].item()
                    ans_len = answer_length[i].item()
                    # student: starts at index qs_len-1 (predicts qs_len+1th token, i.e., first answer token) -> predict ans_len tokens
                    student_slice = student_logits[i, qs_len-1:qs_len+ans_len-1, :]
                    # teacher: last ans_len tokens after context
                    teacher_slice = teacher_logits[i, ctx_len+qs_len-1:ctx_len+qs_len+ans_len-1, :]
                    assert student_slice.shape == teacher_slice.shape
                    aligned_student_logits.append(student_slice)
                    aligned_teacher_logits.append(teacher_slice)
            elif context_length is not None:
                # No answer_length, align with qa_pairs
                for i in range(batch_size):
                    ctx_len = context_length[i].item()
                    # Consider padding effect on input length (student_seq_len > qa_len)
                    qa_len = teacher_seq_len - ctx_len
                    if qa_len > student_seq_len:
                        qa_len = student_seq_len
                    student_slice = student_logits[i, :qa_len, :]
                    teacher_slice = teacher_logits[i, ctx_len-1:ctx_len+qa_len-1, :]
                    assert student_slice.shape == teacher_slice.shape
                    aligned_student_logits.append(student_slice)
                    aligned_teacher_logits.append(teacher_slice)
            else:
                # Simple alignment: assume context is first, target is rest
                aligned_student_logits = student_logits[:, :-1, :]
                aligned_teacher_logits = teacher_logits[:, -(student_logits.size(1)-1):, :].to(student_logits.device)

            if isinstance(aligned_student_logits, list):
                 # pad to max length in batch
                student_logits_aligned = rnn_utils.pad_sequence(aligned_student_logits, batch_first=True, padding_value=0)
                teacher_logits_aligned = rnn_utils.pad_sequence(aligned_teacher_logits, batch_first=True, padding_value=0).to(student_logits.device)
            else:
                student_logits_aligned = aligned_student_logits
                teacher_logits_aligned = aligned_teacher_logits

            loss = self.distillation_loss(student_logits_aligned, teacher_logits_aligned)
            return (loss, student_outputs) if return_outputs else loss
