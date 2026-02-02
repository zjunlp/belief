#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2025/10/03 19:51:41
@Author  :   haoming
@Version :   1.0
'''

import json
from typing import Dict, List, Any, Optional
from datasets import Dataset
from transformers import AutoTokenizer
from trl import apply_chat_template
import torch
import os
from typing import Optional
import random
import copy
from torch.utils.data import Dataset as TorchDataset

class SimpleQASFTDataset:
    """
    Dataset for supervised fine-tuning (SFT) on question-answer pairs using TRL.
    Follows the official TRL dataset format with prompt and completion structure.
    """
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_seq_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load and format data
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        if not isinstance(examples, list) or not all(
            isinstance(d, dict) and ('problem' in d or 'question' or 'original_problem' in d) and ('answer' in d or 'golden_answer' in d or 'answers' or 'original_answer' in d) for d in examples
        ):
            raise ValueError("The JSON file must be a list of dictionaries with 'problem' and 'answer' (or 'golden_answer' or 'answers') keys.")

        # Build dataset dict following TRL format
        dataset_dict = {
            "prompt": [],
            "completion": []
        }
        
        for example in examples:
            problem = example.get('problem') or example.get('question') or example.get('original_problem')
            answer = example.get('answer') or example.get('golden_answer') or example.get('answers') or example.get('original_answer')
            if isinstance(answer, list):
                answer = answer[0]
            elif isinstance(answer, str):
                answer = answer
            else:
                raise ValueError("The answer must be a string or a list of strings.")
            # mix neighbor knowledge
            dataset_dict["prompt"].append([{"role": "user", "content": problem}])
            dataset_dict["completion"].append([{"role": "assistant", "content": answer}])

        # Create HuggingFace Dataset and apply chat template
        self.dataset = Dataset.from_dict(dataset_dict)
        # self.dataset = self.dataset.map(apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer, "chat_template": self.tokenizer.chat_template})
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class MixedSFTDatasetEXP:
    def __init__(self, qa_file_path, doc_file_path, tokenizer, max_seq_length=4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = -100  # PyTorch CrossEntropyLoss default ignore index
        
        # 1. Load data
        data = []
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                data.append({"payload": item, "type": "qa"})
        
        if doc_file_path:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    data.append({"payload": item, "type": "doc"})
        
        raw_dataset = Dataset.from_list(data)
        
        # 2. Core processing
        self.dataset = raw_dataset.map(
            self._process_row,
            remove_columns=raw_dataset.column_names,
            num_proc=4,
            desc="Tokenizing and Creating Labels"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _process_row(self, example):
        payload = example["payload"]
        input_ids = []
        labels = []
        
        # ================= QA data processing logic =================
        if example["type"] == "qa":
            problem = payload.get('problem') or payload.get('question')
            answer = payload.get('answer') or payload.get('golden_answer')
            if isinstance(answer, list): answer = answer[0]
            
            # 1. Build complete conversation
            full_messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer}
            ]
            
            # 2. Build Prompt part (excluding final answer)
            prompt_messages = [
                {"role": "user", "content": problem}
            ]
            
            # 3. Tokenize separately
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages, 
                tokenize=True, 
                add_generation_prompt=True  
            )
            
            full_ids = self.tokenizer.apply_chat_template(
                full_messages, 
                tokenize=True, 
                add_generation_prompt=False
            )
            
            # 4. Truncation (Manual Truncation)
            if len(full_ids) > self.max_seq_length:
                full_ids = full_ids[:self.max_seq_length]
            
            # 5. Generate Labels
            labels = copy.deepcopy(full_ids)
            prompt_len = len(prompt_ids)
            
            if prompt_len > len(full_ids):
                # Prompt longer than truncated sequence
                labels = [self.ignore_index] * len(full_ids)
            else:
                for i in range(prompt_len):
                    labels[i] = self.ignore_index
            
            input_ids = full_ids

        # ================= Doc data processing logic =================
        else:
            text = payload.get('text')
            if text is None or text.strip() == "":
                raise ValueError("Document text is empty or None.")
            
            # 1. Directly Encode
            input_ids = self.tokenizer.encode(
                text, 
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length - 1 # Reserve EOS position
            )
            
            # 2. Manually add EOS
            if not input_ids or input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
            
            labels = copy.deepcopy(input_ids)
            
        # ================= Truncation Only (No Padding) =================
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

        return {
            "input_ids": input_ids,
            "labels": labels
        }

class MixedSFTDataset:
    def __init__(self, qa_file_path, doc_file_path, tokenizer, max_seq_length=4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = -100  # PyTorch CrossEntropyLoss default ignore index
        
        # 1. Load data
        data = []
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                data.append({"payload": item, "type": "qa"})
        
        if doc_file_path:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    data.append({"payload": item, "type": "doc"})
        
        print("show 3 example:")
        for item in data[-3:]:
            print(item)

        raw_dataset = Dataset.from_list(data)
        
        # 2. Core processing
        self.dataset = raw_dataset.map(
            self._process_row,
            remove_columns=raw_dataset.column_names,
            num_proc=4,
            desc="Tokenizing and Creating Labels"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _process_row(self, example):
        payload = example["payload"]
        input_ids = []
        labels = []
        
        # ================= QA data processing logic =================
        if example["type"] == "qa":
            problem = payload.get('problem') or payload.get('question')
            answer = payload.get('answer') or payload.get('golden_answer')
            if isinstance(answer, list): answer = answer[0]
            
            # 1. Build complete conversation
            full_messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer}
            ]
            
            # 2. Build Prompt part (excluding final answer)
            prompt_messages = [
                {"role": "user", "content": problem}
            ]
            
            # 3. Tokenize separately
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages, 
                tokenize=True, 
                add_generation_prompt=True  
            )
            
            full_ids = self.tokenizer.apply_chat_template(
                full_messages, 
                tokenize=True, 
                add_generation_prompt=False
            )
            
            # 4. Truncation (Manual Truncation)
            if len(full_ids) > self.max_seq_length:
                full_ids = full_ids[:self.max_seq_length]
            
            # 5. Generate Labels
            labels = copy.deepcopy(full_ids)
            prompt_len = len(prompt_ids)
            
            if prompt_len > len(full_ids):
                # Prompt longer than truncated sequence
                labels = [self.ignore_index] * len(full_ids)
            else:
                for i in range(prompt_len):
                    labels[i] = self.ignore_index
            
            input_ids = full_ids

        # ================= Doc data processing logic =================
        else:
            text = payload.get('text')
            if text is None or text.strip() == "":
                raise ValueError("Document text is empty or None.")
            
            # 1. Directly Encode
            input_ids = self.tokenizer.encode(
                text, 
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length - 1 # Reserve EOS position
            )
            
            # 2. Manually add EOS
            if not input_ids or input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
            
            labels = copy.deepcopy(input_ids)
            
        # ================= Padding & Attention Mask =================
        # padding/truncation to max_seq_length
        seq_len = len(input_ids)
        padding_len = self.max_seq_length - seq_len
        attention_mask = [1] * seq_len
        
        if padding_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            labels = labels + [self.ignore_index] * padding_len
            attention_mask = attention_mask + [0] * padding_len
        else:
            # Should be handled by truncation above, but safe guard
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class KLDivergenceDataset:
    """
    Dataset for supervised fine-tuning with KL divergence regularization.
    Similar to SimpleQASFTDataset but optimized for KL training where we need
    to compute both base model and fine-tuned model logits on the same inputs.
    """
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_seq_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load and format data
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        if not isinstance(examples, list) or not all(
            isinstance(d, dict) and ('problem' in d or 'question' in d or 'original_problem' in d) and ('answer' in d or 'golden_answer' in d or 'answers' in d or 'original_answer' in d) for d in examples
        ):
            raise ValueError("The JSON file must be a list of dictionaries with 'problem' and 'answer' (or 'golden_answer' or 'answers') keys.")

        # Build dataset dict following TRL format
        dataset_dict = {
            "prompt": [],
            "completion": []
        }
        
        for example in examples:
            problem = example.get('problem') or example.get('question') or example.get('original_problem')
            answer = example.get('answer') or example.get('golden_answer') or example.get('answers') or example.get('original_answer')
            if isinstance(answer, list):
                answer = answer[0]
            elif isinstance(answer, str):
                answer = answer
            else:
                raise ValueError("The answer must be a string or a list of strings.")
            dataset_dict["prompt"].append([{"role": "user", "content": problem}])
            dataset_dict["completion"].append([{"role": "assistant", "content": answer}])

        # Create HuggingFace Dataset and apply chat template
        self.dataset = Dataset.from_dict(dataset_dict)
        self.dataset = self.dataset.map(apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer, "chat_template": self.tokenizer.chat_template})
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class BeliefAugmentedSFTDataset:
    """
    Dataset that prepends neighbor/belief conversations ahead of the original question.
    Only the final original answer is supervised to isolate the impact of noisy beliefs.
    """

    META_SETTING_TO_KEY = {
        "weak_belief": "weak_belief_msgs",
        "incorrect_belief": "incorrect_belief_msgs",
        "reinforce_belief": "reinforce_belief_msgs",
        "consensus_belief": "consensus_msgs",
        "unconsensus_belief": "unconsensus_msgs",
    }

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 4096,
        belief_setting: str = "wrong_belief_entity",
        max_belief_rounds: int = 3,
        system_prompt: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.belief_setting = belief_setting
        self.max_belief_rounds = max(1, max_belief_rounds)
        self.system_prompt = system_prompt

        valid_settings = {"wrong_belief_entity"} | set(self.META_SETTING_TO_KEY.keys())
        if self.belief_setting not in valid_settings:
            raise ValueError(f"belief_setting must be one of {valid_settings}")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("The JSON file must contain a list of belief examples.")

        dataset_dict = {"prompt": [], "completion": []}

        for example in raw_data:
            problem = example.get("original_problem") or example.get("problem") or example.get("question")
            answer = example.get("original_answer") or example.get("answer") or example.get("golden_answer")

            if problem is None or answer is None:
                continue

            history_rounds = self._extract_history_rounds(example)
            prompt_messages: List[Dict[str, str]] = []

            if self.system_prompt:
                prompt_messages.append({"role": "system", "content": self.system_prompt})

            for user_turn, assistant_turn in history_rounds:
                prompt_messages.append({"role": "user", "content": user_turn})
                prompt_messages.append({"role": "assistant", "content": assistant_turn})

            prompt_messages.append({"role": "user", "content": problem})

            dataset_dict["prompt"].append(prompt_messages)
            dataset_dict["completion"].append([{"role": "assistant", "content": answer}])

        self.dataset = Dataset.from_dict(dataset_dict)
        self.dataset = self.dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer, "chat_template": self.tokenizer.chat_template}
        )

    def _extract_history_rounds(self, example: Dict[str, Any]) -> List[Any]:
        if self.belief_setting == "wrong_belief_entity":
            return self._extract_wrong_entity_rounds(example)

        meta_key = self.META_SETTING_TO_KEY[self.belief_setting]
        experiment_meta = example.get("experiment_meta", {})
        messages = experiment_meta.get(meta_key, [])
        return self._messages_to_rounds(messages)

    def _extract_wrong_entity_rounds(self, example: Dict[str, Any]) -> List[Any]:
        rounds = []
        for neighbor in example.get("neighbor_questions", []):
            question = neighbor.get("question")
            wrong_entity = neighbor.get("wrong_belief_entity")
            if not question or not wrong_entity:
                continue
            rounds.append((question, wrong_entity))
            if len(rounds) >= self.max_belief_rounds:
                break
        return rounds

    def _messages_to_rounds(self, messages: List[Dict[str, str]]) -> List[Any]:
        rounds = []
        pending_question: Optional[str] = None
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if not content:
                continue
            if role == "user":
                pending_question = content
            elif role == "assistant" and pending_question:
                rounds.append((pending_question, content))
                pending_question = None
                if len(rounds) >= self.max_belief_rounds:
                    break
        return rounds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class DPODataset:
    """
    Dataset for Direct Preference Optimization (DPO) training using TRL.
    
    Expected JSON format:
    [
        {   "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [{"role": "assistant", "content": "It is blue."}],
            "rejected": [{"role": "assistant", "content": "It is green."}]
        },
        ...
    ]
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if not isinstance(raw_data, list):
            raise ValueError("JSON file must contain a list of examples")
        
        dataset_dict = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        for item in raw_data:
            dataset_dict["prompt"].append([{"role": "user", "content": item["prompt"]}])
            dataset_dict["chosen"].append([{"role": "assistant", "content": item["chosen"]}])
            dataset_dict["rejected"].append([{"role": "assistant", "content": item["rejected"]}])
        
        self.dataset = Dataset.from_dict(dataset_dict)
        self.dataset = self.dataset.map(apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer, "chat_template": self.tokenizer.chat_template})
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class GRPODataset:
    """
    Dataset for Group Relative Policy Optimization (GRPO) training using TRL.
    
    GRPO is similar to RLHF but uses group-relative advantages.
    The dataset format is similar to standard prompts for generation.
    
    Expected JSON format:
    [
        {
            "problem": "1+1=?",
            "answer": "2",
            "reasoning": "optional reasoning process"  # Optional field
        }
        ...
    ]
    
    If use_answer_tags=True:
    - With reasoning: formats as "<think>reasoning</think> <answer>answer</answer>"
    - Without reasoning: formats as "<think></think> <answer>answer</answer>" (empty think tags)
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 4096,
        use_system_prompt: bool = True,
        use_answer_tags: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_system_prompt = use_system_prompt
        self.use_answer_tags = use_answer_tags
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if not isinstance(raw_data, list):
            raise ValueError("JSON file must contain a list of examples")
        
        dataset_dict = {
            "prompt": [],  # Prompts for generation
            "solution": [],
            "question": []
        }

        for item in raw_data:
            # Build prompt with optional system message
            dataset_dict["question"].append(item["question"])
            prompt_messages = []
            if self.use_system_prompt:
                system_msg = """A conversation between User and Assistant. 
                The user asks a question, and the Assistant solves it. 
                The assistant first thinks about the reasoning process in the mind, and then provides the user with the final answer. 
                The format that must be followed is: <think> reasoning process here </think> <answer> final answer here </answer>"""

                prompt_messages.append({"role": "system", "content": system_msg})
            
            nqs = item.get("neighbor_questions", [])
            for nq in nqs:
                prompt_messages.append({"role": "user", "content": nq.get("question", "")})
                prompt_messages.append({"role": "assistant", "content": nq.get("correct_answer", "")})
             # If it's a list, concatenate prompt in specified format
            context = item.get("support", [])
            if isinstance(context, list):
                # Initialize formatted result string
                formatted_str = ""
                # Iterate list, concatenate in specified format by index (starting from 1)
                for idx, content in enumerate(context, start=1):
                    # Concatenate single element format (preserve newlines, consistent with example)
                    formatted_str += f"<support_fact{idx}>\n{content}\n</support_fact{idx}>\n"
                    # Remove trailing newlines (optional, adjust as needed)
                context = formatted_str.rstrip("\n")
            content = context + "\n" + item["question"] if context else item["question"]
            prompt_messages.append({"role": "user", "content": content})
            dataset_dict["prompt"].append(prompt_messages)
            
            # Support both 'answer' and 'golden_answer' keys
            answer = item.get("answer") or item.get("golden_answer", "")
            
            # Optionally wrap answer in tags for training examples
            if self.use_answer_tags:
                # Check if reasoning/thinking is available in data
                reasoning = item.get("reasoning") or item.get("think", "")
                
                # Always use both <think> and <answer> tags to match system prompt format
                # Use empty <think> if no reasoning is provided
                formatted_response = f"<think>{reasoning}</think> <answer>{answer}</answer>"
                
                answer = formatted_response
            
            dataset_dict["solution"].append([{"role": "assistant", "content": answer}])
        
        # GRPO expects prompt and solution as message lists (not tokenized strings)
        self.dataset = Dataset.from_dict(dataset_dict)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class ContextInjectionDataset:
    """
    Dataset for CONTEXT-INJECTION training that supports context injection via knowledge distillation.
    
    This dataset prepares data in two formats:
    1. With context: [BOS] + context + question/answer (for teacher model)
    2. Without context: [BOS] + question/answer (for student model)
    
    Expected JSON format:
    [
        {
            "context": "Some contextual information...",
            "problem": "Question based on context",
            "answer": "Answer to the question"
        },
        ...
    ]
    
    Or with pre-generated QA pairs:
    [
        {
            "context": "Some contextual information...",
            "qa_pairs": [
                {"question": "Q1?", "answer": "A1"},
                {"question": "Q2?", "answer": "A2"}
            ]
        },
        ...
    ]
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 4096,
        use_chat_template: bool = True,
        context_key: str = "support",
        problem_key: str = "question",
        answer_key: str = "answer",
        qa_pairs_key: str = "qa_pairs",
        only_use_answers: bool = True,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_chat_template = use_chat_template
        self.context_key = context_key
        self.problem_key = problem_key
        self.answer_key = answer_key
        self.qa_pairs_key = qa_pairs_key
        self.only_use_answers = only_use_answers
        self.max_samples = max_samples
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if not isinstance(raw_data, list):
            raise ValueError("JSON file must contain a list of examples")
        # --- New logic: sample if max_samples is specified ---
        if max_samples is not None:
            random.seed(42)  # Fixed seed for reproducible sampling (can customize)
            raw_data = random.sample(raw_data, k=self.max_samples)  # Random sample k items without replacement
        # Process data
        if self.only_use_answers:
            self.examples = self._process_data_only_answers(raw_data)
            # Add column_names for TRL compatibility
            self.column_names = ["input_ids", "attention_mask", "context_input_ids", "context_attention_mask", "context_length","question_length","answer_length"]
        else:
            self.examples = self._process_data(raw_data)
            self.column_names = ["input_ids", "attention_mask", "context_input_ids", "context_attention_mask", "context_length"]
        # Convert to HuggingFace Dataset for TRL compatibility
        self.dataset = Dataset.from_list(self.examples)
        print(f"Loaded {len(self.dataset)} examples for Context-Injection Dataset.")        
    def _process_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Process raw data into training examples with tokenization.
        Expands qa_pairs if present, otherwise uses problem/answer format.
        """
        examples = []
        
        for item in raw_data:
            context = item.get(self.context_key, [])
            
            # Check if qa_pairs format
            if self.qa_pairs_key in item:
                qa_pairs = item[self.qa_pairs_key]
                for qa in qa_pairs:
                    problem = qa.get("question", qa.get(self.problem_key, ""))
                    answer = qa.get("answer", qa.get(self.answer_key, ""))
                    processed_example = self._tokenize_example(context, problem, answer)
                    if processed_example:
                        examples.append(processed_example)
            # Standard problem/answer format
            elif self.problem_key in item and self.answer_key in item:
                problem = item[self.problem_key]
                answer = item[self.answer_key]
                processed_example = self._tokenize_example(context, problem, answer)
                if processed_example:
                    examples.append(processed_example)
            else:
                # Skip malformed examples
                continue
        
        return examples
    def _process_data_only_answers(self, raw_data: List[Dict]) -> List[Dict]:
        examples = []
        
        for item in raw_data:
            context = item.get(self.context_key, "")
            
            # Check if qa_pairs format
            if self.qa_pairs_key in item:
                qa_pairs = item[self.qa_pairs_key]
                for qa in qa_pairs:
                    problem = qa.get("question", qa.get(self.problem_key, ""))
                    answer = qa.get("answer", qa.get(self.answer_key, ""))
                    processed_example = self._tokenize_example_only_answers(context, problem, answer)
                    if processed_example:
                        examples.append(processed_example)
            # Standard problem/answer format
            elif self.problem_key in item and self.answer_key in item:
                problem = item[self.problem_key]
                answer = item[self.answer_key]
                processed_example = self._tokenize_example_only_answers(context, problem, answer)
                if processed_example:
                    examples.append(processed_example)
            else:
                # Skip malformed examples
                continue
        return examples
    def _tokenize_example(self, context: List[str], problem: str, answer: str) -> Optional[Dict]:
        """Tokenize a single example and return processed data."""
        try:
            # Create QA text
            qa_text = self._create_chat_messages(problem, answer)
            # If it's a list, concatenate prompt in specified format
            if isinstance(context, list):
                # Initialize formatted result string
                formatted_str = ""
                # Iterate list, concatenate in specified format by index (starting from 1)
                for idx, content in enumerate(context, start=1):
                    # Concatenate single element format (preserve newlines, consistent with example)
                    formatted_str += f"<support_fact{idx}>\n{content}\n</support_fact{idx}>\n"
                    # Remove trailing newlines (optional, adjust as needed)
                context = formatted_str.rstrip("\n")
            # Tokenize components
            context_ids = self._tokenize_text(context, add_special_tokens=False)
            qa_ids = self._tokenize_text(qa_text, add_special_tokens=False)
            
            # Create student input (no context)
            student_input_ids = qa_ids
            
            # Create teacher input (with context)
            teacher_input_ids = torch.cat([context_ids, qa_ids])
            
            # Truncate if needed
            if len(student_input_ids) > self.max_seq_length:
                student_input_ids = student_input_ids[:self.max_seq_length]
            
            if len(teacher_input_ids) > self.max_seq_length:
                # Prioritize keeping the QA part, truncate context if needed
                context_budget = self.max_seq_length - len(qa_ids)
                if context_budget > 0:
                    context_ids = context_ids[:context_budget]
                    teacher_input_ids = torch.cat([context_ids, qa_ids])
                else:
                    # If even QA is too long, just use student input
                    teacher_input_ids = student_input_ids
                    context_ids = torch.tensor([], dtype=torch.long)
            
            context_length = len(context_ids)
            
            return {
                "input_ids": student_input_ids,
                "attention_mask": torch.ones_like(student_input_ids),
                "context_input_ids": teacher_input_ids,
                "context_attention_mask": torch.ones_like(teacher_input_ids),
                "context_length": context_length,
            }
        except Exception as e:
            print(f"Error tokenizing example: {e}")
            return None
    def _tokenize_example_only_answers(self, context: List[str], problem: str, answer: str) -> Optional[Dict]:
        """Tokenize a single example and return processed data."""
        try:
            if isinstance(context, list):
                # Initialize formatted result string
                formatted_str = ""
                # Iterate list, concatenate in specified format by index (starting from 1)
                for idx, content in enumerate(context, start=1):
                    # Concatenate single element format (preserve newlines, consistent with example)
                    formatted_str += f"<support_fact{idx}>\n{content}\n</support_fact{idx}>\n"
                    # Remove trailing newlines (optional, adjust as needed)
                context = formatted_str.rstrip("\n")
            # Tokenize components
            context_ids = self._tokenize_text(context, add_special_tokens=False)
            # apply_chat_template for problem and answer 
            if self.use_chat_template:
                question_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": problem}],
                    tokenize=False,
                    add_generation_prompt=False
                )
                answer_text = self.tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": answer}],
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                question_text = problem
                answer_text = answer
            question_ids = self._tokenize_text(question_text, add_special_tokens=False)
            answer_ids = self._tokenize_text(answer_text, add_special_tokens=False)
            qa_ids = torch.cat([question_ids, answer_ids])
            # Create student input (no context)
            student_input_ids = qa_ids
            question_length = len(question_ids)
            answer_length = len(answer_ids)
            context_length = len(context_ids)
            # Create teacher input (with context)
            teacher_input_ids = torch.cat([context_ids, qa_ids])
            # Truncate if needed
            if len(student_input_ids) > self.max_seq_length:
                student_input_ids = student_input_ids[:self.max_seq_length]
                if question_length >= self.max_seq_length:
                    question_length = self.max_seq_length
                    answer_length = 0
                else:
                    answer_length = min(answer_length, self.max_seq_length - question_length)
            if context_length + len(qa_ids) > self.max_seq_length:
                # Prioritize keeping the QA part, truncate context if needed
                context_budget = self.max_seq_length - len(qa_ids)
                if context_budget > 0:
                    context_ids = context_ids[:context_budget]
                    teacher_input_ids = torch.cat([context_ids, qa_ids])
                    context_length = len(context_ids)
                else:
                    # If even QA is too long, just use student input
                    teacher_input_ids = student_input_ids
                    context_ids = torch.tensor([], dtype=torch.long)
            context_length = len(context_ids)            
            return {
                "input_ids": student_input_ids,
                "attention_mask": torch.ones_like(student_input_ids),
                "context_input_ids": teacher_input_ids,
                "context_attention_mask": torch.ones_like(teacher_input_ids),
                "context_length": context_length,
                "question_length":question_length,
                "answer_length": answer_length,
            }
        except Exception as e:
            print(f"Error tokenizing example: {e}")
            return None
    def _tokenize_text(self, text: str, add_special_tokens: bool = False) -> torch.Tensor:
        """Tokenize text and return as tensor."""
        ids = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=self.max_seq_length
        ).input_ids
        return torch.tensor(ids, dtype=torch.long)
    
    def _create_chat_messages(self, problem: str, answer: str) -> str:
        """Create chat-formatted messages."""
        if self.use_chat_template:
            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            return f"{problem}\n{answer}"
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Return preprocessed data from HuggingFace Dataset."""
        return self.dataset[idx]

class ContextInjectionDataCollator:
    """
    Custom data collator for CONTEXT-INJECTION that handles batching of dual-input format.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, padding: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch with proper padding.
        """
        # Extract different input types
        student_inputs = [f["input_ids"] for f in features]
        teacher_inputs = [f["context_input_ids"] for f in features]
        context_lengths = torch.tensor([f["context_length"] for f in features], dtype=torch.long)
        answer_lengths = None
        question_lengths = None
        if "answer_length" in features[0]:
            answer_lengths = torch.tensor([f["answer_length"] for f in features], dtype=torch.long)
        if "question_length" in features[0]:
            question_lengths = torch.tensor([f["question_length"] for f in features], dtype=torch.long)
        # Pad sequences
        student_input_ids = self._pad_sequences(student_inputs)
        teacher_input_ids = self._pad_sequences(teacher_inputs)
        
        # Create attention masks
        student_attention_mask = (student_input_ids != self.pad_token_id).long()
        teacher_attention_mask = (teacher_input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": student_input_ids,
            "attention_mask": student_attention_mask,
            "context_input_ids": teacher_input_ids,
            "context_attention_mask": teacher_attention_mask,
            "context_length": context_lengths,
            "question_length":question_lengths,
            "answer_length": answer_lengths
        }
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to same length."""
        # Convert lists to tensors if needed
        tensor_sequences = []
        for seq in sequences:
            if isinstance(seq, list):
                tensor_sequences.append(torch.tensor(seq, dtype=torch.long))
            else:
                tensor_sequences.append(seq)
        
        max_len = max(len(seq) for seq in tensor_sequences)
        padded = []
        
        for seq in tensor_sequences:
            padding_length = max_len - len(seq)
            if padding_length > 0:
                # Pad on the right
                padded_seq = torch.cat([
                    seq,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
            else:
                padded_seq = seq
            padded.append(padded_seq)
        
        return torch.stack(padded)

class ContextDistillationDataset:
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 4096,
        use_chat_template: bool = True,
        max_samples: int = 7000,
    ):
        max_samples = None
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_chat_template = use_chat_template
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        if max_samples is not None:
            raw_data = raw_data[:max_samples]
            
        examples = []
        for item in raw_data:
            text = item.get("text", "")
            question = item.get("question", "")
            answer = item.get("answer", "")
            ex = self._tokenize_example(text, question, answer)
            if ex:
                examples.append(ex)
                
        self.dataset = Dataset.from_list(examples)
        self.column_names = ["student_input_ids", "student_attention_mask", "teacher_input_ids", "teacher_attention_mask", "student_question_length", "teacher_question_length", "answer_length"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    def _tokenize_example(self, text: str, question: str, answer: str) -> Optional[Dict]:
        try:
            # 1. Build message list
            student_messages = [{"role": "user", "content": (text + "\n" + question) if text else question}]
            teacher_messages = [{"role": "user", "content": question}]
            answer_message = {"role": "assistant", "content": answer}

            # 2. Student Tokenization
            student_prompt_ids = self.tokenizer.apply_chat_template(
                student_messages, tokenize=True, add_generation_prompt=True
            )
            student_full_ids = self.tokenizer.apply_chat_template(
                student_messages + [answer_message], tokenize=True
            )
            
            # Validation
            if len(student_full_ids) <= len(student_prompt_ids):
                raise ValueError("Student generated too short answer.")
                
            # Extract Answer IDs
            answer_ids = student_full_ids[len(student_prompt_ids):]

            # 3. Teacher Tokenization (force align Answer)
            teacher_prompt_ids = self.tokenizer.apply_chat_template(
                teacher_messages, tokenize=True, add_generation_prompt=True
            )
            teacher_full_ids = teacher_prompt_ids + answer_ids

            # 4. Convert to Tensor
            s_ids = torch.tensor(student_full_ids, dtype=torch.long)
            t_ids = torch.tensor(teacher_full_ids, dtype=torch.long)
            
            s_q_len = len(student_prompt_ids)
            t_q_len = len(teacher_prompt_ids)
            a_len = min(len(answer_ids), self.max_seq_length)

            
            # --- Process Student ---
            if len(s_ids) > self.max_seq_length:
                # Calculate how many tokens to discard
                num_dropped = len(s_ids) - self.max_seq_length
                
                # Keep last max_seq_length tokens (keep right side)
                s_ids = s_ids[-self.max_seq_length:]
                
                # Because left side is cut, Prompt length is shorter, boundary shifts left
                s_q_len = max(0, s_q_len - num_dropped)

            # --- Process Teacher ---
            if len(t_ids) > self.max_seq_length:
                num_dropped = len(t_ids) - self.max_seq_length
                t_ids = t_ids[-self.max_seq_length:]
                t_q_len = max(0, t_q_len - num_dropped)

            return {
                "student_input_ids": s_ids,
                "student_attention_mask": torch.ones_like(s_ids),
                "teacher_input_ids": t_ids,
                "teacher_attention_mask": torch.ones_like(t_ids),
                "student_question_length": s_q_len, 
                "teacher_question_length": t_q_len,
                "answer_length": a_len,
            }

        except Exception as e:
            print(f"Error tokenizing example: {e}")
            return None

class ContextDistillationDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, padding: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        student_inputs = [f["student_input_ids"] for f in features]
        teacher_inputs = [f["teacher_input_ids"] for f in features]
        student_question_lengths = torch.tensor([f["student_question_length"] for f in features], dtype=torch.long)
        teacher_question_lengths = torch.tensor([f["teacher_question_length"] for f in features], dtype=torch.long)
        answer_lengths = torch.tensor([f["answer_length"] for f in features], dtype=torch.long) if "answer_length" in features[0] else None
        student_input_ids = self._pad_sequences(student_inputs)
        teacher_input_ids = self._pad_sequences(teacher_inputs)
        student_attention_mask = (student_input_ids != self.pad_token_id).long()
        teacher_attention_mask = (teacher_input_ids != self.pad_token_id).long()
        result = {
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "student_question_length": student_question_lengths,
            "teacher_question_length": teacher_question_lengths,
        }
        if answer_lengths is not None:
            result["answer_length"] = answer_lengths
        else:
            raise ValueError("Answer lengths are required for distillation.")
        return result
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to same length."""
        # Convert lists to tensors if needed
        tensor_sequences = []
        for seq in sequences:
            if isinstance(seq, list):
                tensor_sequences.append(torch.tensor(seq, dtype=torch.long))
            else:
                tensor_sequences.append(seq)
        
        max_len = max(len(seq) for seq in tensor_sequences)
        padded = []
        
        for seq in tensor_sequences:
            padding_length = max_len - len(seq)
            if padding_length > 0:
                # Pad on the right
                padded_seq = torch.cat([
                    seq,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
            else:
                padded_seq = seq
            padded.append(padded_seq)
        
        return torch.stack(padded)

class ContextDistillationWithKLDataset(TorchDataset):
    def __init__(self, distill_dataset: TorchDataset, kl_dataset: TorchDataset, kl_sampling: str = "random"):
        self.distill_dataset = distill_dataset
        self.kl_dataset = kl_dataset
        self.distill_len = len(self.distill_dataset)
        self.kl_len = len(self.kl_dataset)
        self._len = max(self.distill_len, self.kl_len)
        self.kl_sampling = "paired" if kl_sampling == "sequential" else kl_sampling

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        distill_idx = idx % self.distill_len
        distill_item = self.distill_dataset[distill_idx]

        if self.kl_sampling == "paired":
            kl_idx = idx % self.kl_len
        elif self.kl_sampling == "random":
            kl_idx = random.randrange(self.kl_len)
        else:
            raise ValueError(f"Unknown kl_sampling: {self.kl_sampling}. Expected 'random', 'paired', or 'sequential'.")

        kl_item = self.kl_dataset[kl_idx]
        return distill_item, kl_item


class ContextDistillationWithKLDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, distill_collator: Optional[ContextDistillationDataCollator] = None):
        self.tokenizer = tokenizer
        self.distill_collator = distill_collator if distill_collator is not None else ContextDistillationDataCollator(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: List[Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        distill_features = [f[0] for f in features]
        kl_features = [f[1] for f in features]

        cd_batch = self.distill_collator(distill_features)

        kl_input_ids = [x["input_ids"] for x in kl_features]
        kl_labels = [x["labels"] for x in kl_features]

        kl_input_ids_t = self._pad_2d(kl_input_ids, self.pad_token_id)
        kl_labels_t = self._pad_2d(kl_labels, -100)
        kl_attn = (kl_input_ids_t != self.pad_token_id).long()

        return {
            "context_distillation": cd_batch,
            "kl": {
                "input_ids": kl_input_ids_t,
                "attention_mask": kl_attn,
                "labels": kl_labels_t,
            },
        }

    def _pad_2d(self, seqs: List[Any], pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs) if len(seqs) > 0 else 0
        out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            if isinstance(s, torch.Tensor):
                s = s.tolist()
            if len(s) == 0:
                continue
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out


class ContextDistillationWithKLSFTDataset(TorchDataset):
    def __init__(
        self,
        distill_dataset: TorchDataset,
        kl_dataset: TorchDataset,
        sft_dataset: TorchDataset,
        kl_sampling: str = "random",
        sft_sampling: str = "random",
    ):
        self.distill_dataset = distill_dataset
        self.kl_dataset = kl_dataset
        self.sft_dataset = sft_dataset
        self.distill_len = len(self.distill_dataset)
        self.kl_len = len(self.kl_dataset)
        self.sft_len = len(self.sft_dataset)
        self._len = max(self.distill_len, self.kl_len, self.sft_len)
        self.kl_sampling = "paired" if kl_sampling == "sequential" else kl_sampling
        self.sft_sampling = "paired" if sft_sampling == "sequential" else sft_sampling

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        distill_idx = idx % self.distill_len
        distill_item = self.distill_dataset[distill_idx]

        if self.kl_sampling == "paired":
            kl_idx = idx % self.kl_len
        elif self.kl_sampling == "random":
            kl_idx = random.randrange(self.kl_len)
        else:
            raise ValueError(f"Unknown kl_sampling: {self.kl_sampling}. Expected 'random', 'paired', or 'sequential'.")

        if self.sft_sampling == "paired":
            sft_idx = idx % self.sft_len
        elif self.sft_sampling == "random":
            sft_idx = random.randrange(self.sft_len)
        else:
            raise ValueError(f"Unknown sft_sampling: {self.sft_sampling}. Expected 'random', 'paired', or 'sequential'.")

        kl_item = self.kl_dataset[kl_idx]
        sft_item = self.sft_dataset[sft_idx]
        return distill_item, kl_item, sft_item


class ContextDistillationWithKLSFTDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, distill_collator: Optional[ContextDistillationDataCollator] = None):
        self.tokenizer = tokenizer
        self.distill_collator = distill_collator if distill_collator is not None else ContextDistillationDataCollator(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: List[Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        distill_features = [f[0] for f in features]
        kl_features = [f[1] for f in features]
        sft_features = [f[2] for f in features]

        cd_batch = self.distill_collator(distill_features)

        kl_input_ids = [x["input_ids"] for x in kl_features]
        kl_labels = [x["labels"] for x in kl_features]

        sft_input_ids = [x["input_ids"] for x in sft_features]
        sft_labels = [x["labels"] for x in sft_features]

        kl_input_ids_t = self._pad_2d(kl_input_ids, self.pad_token_id)
        kl_labels_t = self._pad_2d(kl_labels, -100)
        kl_attn = (kl_input_ids_t != self.pad_token_id).long()

        sft_input_ids_t = self._pad_2d(sft_input_ids, self.pad_token_id)
        sft_labels_t = self._pad_2d(sft_labels, -100)
        sft_attn = (sft_input_ids_t != self.pad_token_id).long()

        return {
            "context_distillation": cd_batch,
            "kl": {
                "input_ids": kl_input_ids_t,
                "attention_mask": kl_attn,
                "labels": kl_labels_t,
            },
            "sft": {
                "input_ids": sft_input_ids_t,
                "attention_mask": sft_attn,
                "labels": sft_labels_t,
            },
        }

    def _pad_2d(self, seqs: List[Any], pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs) if len(seqs) > 0 else 0
        out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            if isinstance(s, torch.Tensor):
                s = s.tolist()
            if len(s) == 0:
                continue
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out
