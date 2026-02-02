#!/usr/bin/env python3

import json
import argparse
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Dict, Any, Union
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.utils import load_json, save_json

# --- Prompt Templates ---

ENTITY_EXTRACTION_PROMPT_exp = """Your task is to extract the main answer entity from the Response that directly answers the Question. You will use the expected answer type as a reference to better understand the nature of the answer you should extract.

First, please carefully read the following information:
<Question>
{question}
</Question>
<ExpectedAnswerType>
{expected_answer_type}
</ExpectedAnswerType>
<Response>
{response}
</Response>

When extracting the entity, please follow these principles **strictly**:

- **Direct Entity Rule**
   - If the Response itself is a single entity (a name, number, date, yes/no, or short noun phrase), return it directly.

- **Focused Extraction Rule (MUST EXTRACT)**
   - Otherwise, find the one concise noun phrase or named entity that most directly answers the Question.
   - **PRIORITIZATION:** If the answer entity is clearly present in the Response (even if it is complex or embedded in descriptive text), you **MUST** attempt to extract it.
   - Use the Answer Type only to understand what *type* of entity should be extracted.
   - If multiple entities are present, pick the one that most likely fulfills the Question's intent.

- **Cautious NOT_ATTEMPTED Rule (STRICT LIMITATION)**
   - Only output `NOT_ATTEMPTED` **if and only if** the Response contains explicit phrases indicating the answer is **missing, unknown, or cannot be found within the provided context**.
   - (e.g., *"unknown"*, *"not provided"*, *"not specified"*, *"cannot be determined"*, or *"no information given"*).
   - **DO NOT** output `NOT_ATTEMPTED` if an answer entity is present, regardless of how descriptive the surrounding text is.

- **Contradiction Isolation Rule**
   - If the Response has conflicting information, extract the entity that directly answers the Question and ignore contradictory non-core details. Do not reject the core entity due to incorrect supplements.

- **Output Format**
   - The output **MUST** be **exactly** the extracted entity text (matching case/spacing of the source) or the specific phrase `NOT_ATTEMPTED`.
   - **Absolutely no** punctuation, quotation marks, or explanatory text before or after the output.

Please output the extracted entity now:
"""

def build_extraction_messages(question: str, ans_type: str, response: str) -> List[Dict[str, str]]:
    """Build Chat format Prompt"""
    content = ENTITY_EXTRACTION_PROMPT_exp.format(
        question=question,
        expected_answer_type=ans_type,
        response=response
    )
    return [{"role": "user", "content": content}]


def extract_expected_answer_type(metadata: Dict[str, Any]) -> str:
    """Extract expected_answer_type from metadata deeply"""
    paths = [
        metadata.get("metadata", {}).get("metadata", {}).get("metadata", {}).get("expected_answer_type", ""),
        metadata.get("metadata", {}).get("metadata", {}).get("expected_answer_type", ""),
        metadata.get("metadata", {}).get("expected_answer_type", ""),
        metadata.get("expected_answer_type", "")
    ]
    for path in paths:
        if path:
            return path
    return "default"


class EntityExtractor:
    """
    Streamlined Entity Extractor for Asch and Source Conflict experiments.
    """
    
    def __init__(self, args):
        self.args = args
        self.llm = None
        self.sampling_params = None
        self.lora_request = None

    def initialize_model(self):
        """Initialize the extraction model"""
        print(f"Loading Judge Model: {self.args.judge_model_path}")
        if self.args.lora_path:
            print(f"LoRA enabled: {self.args.lora_path}")
        
        # Extraction should be deterministic
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
        
        if self.args.lora_path:
            self.lora_request = LoRARequest("judge_adapter", 1, self.args.lora_path)
        
        # Optimize vLLM for batch processing performance
        llm_kwargs = {
            "model": self.args.judge_model_path,
            "tensor_parallel_size": self.args.num_gpu,
            "enable_lora": bool(self.args.lora_path),
            "gpu_memory_utilization": getattr(self.args, 'gpu_memory_utilization', 0.9),
            "trust_remote_code": True,
            "max_num_batched_tokens": getattr(self.args, 'max_num_batched_tokens', 8192),  # Increase batch token limit
            "max_num_seqs": getattr(self.args, 'max_num_seqs', 256),  # Increase concurrent sequences
        }
        
        # Add enforce_eager if specified (can help with memory but may be slower)
        if hasattr(self.args, 'enforce_eager') and self.args.enforce_eager:
            llm_kwargs["enforce_eager"] = True
            
        self.llm = LLM(**llm_kwargs)
    
    def _get_response_text(self, raw_response: Any) -> str:
        """
        Helper: Handle various response formats (CoT dict, string, parsed dict).
        """
        if isinstance(raw_response, dict):
            # 1. New CoT format
            if 'final_answer' in raw_response:
                return raw_response['final_answer']
            # 2. Parsed format
            if 'text' in raw_response:
                return raw_response['text']
            # 3. Fallback
            if 'answer' in raw_response:
                return raw_response['answer']
            # 4. JSON dump
            return str(raw_response)
        
        return str(raw_response)

    def _add_task(self, responses: List[Any], source_key: str, result_key: str, 
                  original_question: str, expected_answer_type: str, 
                  data_idx: int, all_conversations: List, request_metadata: List):
        
        for sample_i, raw_resp in enumerate(responses):
            resp_text = self._get_response_text(raw_resp)
            
            if not resp_text or not resp_text.strip():
                resp_text = "NOT_ATTEMPTED"
                
            messages = build_extraction_messages(original_question, expected_answer_type, resp_text)
            all_conversations.append(messages)
            request_metadata.append({
                "type": "standard",
                "data_idx": data_idx,
                "result_key": result_key,
                "sample_idx": sample_i
            })

    def _add_followup_task(self, followup_data: List[Dict], source_key: str, result_key: str, original_question: str, expected_answer_type: str, data_idx: int, all_conversations: List, request_metadata: List):
        
        for sample_i, f_item in enumerate(followup_data):
            resp_text = ""
            
            # Handle standardized followup storage
            if "followup_parsed" in f_item and f_item["followup_parsed"]:
                 resp_text = self._get_response_text(f_item["followup_parsed"])
            elif "followup_response_raw" in f_item:
                resp_text = f_item["followup_response_raw"]
            elif "followup_response" in f_item:
                resp_text = self._get_response_text(f_item["followup_response"])
            
            if not resp_text or not resp_text.strip():
                resp_text = "NOT_ATTEMPTED"

            messages = build_extraction_messages(original_question, expected_answer_type, resp_text)
            all_conversations.append(messages)
            request_metadata.append({
                "type": "followup",
                "data_idx": data_idx,
                "result_key": result_key,
                "sample_idx": sample_i
            })

    def prepare_extraction_tasks(self, data: List[Dict[str, Any]]) -> tuple:
        """
        Prepare extraction tasks by dynamically scanning for Asch/Source keys.
        """
        all_conversations = []
        request_metadata = []
        
        # === Define target experiment names ===
        # These must match the mode in generation scripts
        target_experiment_prefixes = [
            "asch_conflict",      
            "asch_misleading",    
            "source_misleading",  
            "source_conflict"     
        ]

        print(f"Preparing extraction tasks...")
        
        for idx, item in enumerate(tqdm(data, desc="Scanning items")):
            metadata = item.get("metadata", {})
            original_question = metadata.get("original_question", "")
            expected_answer_type = extract_expected_answer_type(metadata)
            
            all_keys = list(item.keys())

            for key in all_keys:
                # -------------------------------------------------
                # 1. Process main responses (starting with resp_)
                # -------------------------------------------------
                if key.startswith("resp_"):
                    # key = "resp_asch_conflict_cfg0_std"
                    # stripped_name = "asch_conflict_cfg0_std"
                    stripped_name = key[len("resp_"):] 
                    
                    if any(stripped_name.startswith(prefix) for prefix in target_experiment_prefixes):
                        # Result Key: extracted_entities_asch_conflict_cfg0_std
                        result_key = f"extracted_entities_{stripped_name}"
                        
                        if result_key not in item:
                            item[result_key] = [None] * len(item[key])
                        
                        self._add_task(
                            item[key], key, result_key,
                            original_question, expected_answer_type, 
                            idx, all_conversations, request_metadata
                        )

                # -------------------------------------------------
                # 2. Process Follow-ups (starting with followup_)
                # -------------------------------------------------
                if self.args.extract_followup and key.startswith("followup_"):
                    # key = "followup_asch_conflict_cfg0_std_lvl1"
                    
                    # [Key fix]: Remove "followup_" prefix
                    # stripped_name = "asch_conflict_cfg0_std_lvl1"
                    stripped_name = key[len("followup_"):]
                    
                    if any(stripped_name.startswith(prefix) for prefix in target_experiment_prefixes):
                        # Result Key: extracted_followup_entities_asch_conflict_cfg0_std_lvl1
                        # (This avoids the double extracted_followup_entities_followup_... issue)
                        result_key = f"extracted_followup_entities_{stripped_name}"
                        
                        if result_key not in item:
                            item[result_key] = [None] * len(item[key])
                            
                        self._add_followup_task(
                            item[key], key, result_key,
                            original_question, expected_answer_type,
                            idx, all_conversations, request_metadata
                        )

        print(f"Total extraction tasks prepared: {len(all_conversations)}")
        return all_conversations, request_metadata
    
    def extract_entities(self, conversations: List[List[Dict[str, str]]], request_metadata: List[Dict[str, Any]] = None) -> List:
        """
        Extract entities with optimized batch processing.
        Uses larger batches and optimized vLLM parameters for better throughput.
        """
        total = len(conversations)
        batch_size = getattr(self.args, 'batch_size', 50000)  # Default batch size
        
        print(f"Running entity extraction with {total} requests (batch size: {batch_size})...")
        print(f"Estimated batches: {(total + batch_size - 1) // batch_size}")
        
        all_outputs = []
        num_batches = (total + batch_size - 1) // batch_size
        
        for batch_idx, batch_start in enumerate(tqdm(range(0, total, batch_size), desc="Processing batches", total=num_batches)):
            batch_end = min(batch_start + batch_size, total)
            batch_conversations = conversations[batch_start:batch_end]
            batch_num = batch_idx + 1
            
            if batch_num % 10 == 0 or batch_num == 1:
                print(f"\nProcessing batch {batch_num}/{num_batches} (items {batch_start}-{batch_end-1} of {total})...")
            
            # Use optimized vLLM chat with larger batches
            # vLLM handles batching internally, so we can pass larger batches
            batch_outputs = self.llm.chat(
                messages=batch_conversations,
                sampling_params=self.sampling_params,
                lora_request=self.lora_request,
                use_tqdm=True  # Disable inner tqdm to avoid nested progress bars
            )
            all_outputs.extend(batch_outputs)
            
            # Optional: Print throughput stats every 10 batches
            if batch_num % 10 == 0:
                processed = batch_end
                print(f"  Progress: {processed}/{total} ({100*processed/total:.1f}%)")
        
        return all_outputs
    
    def process_extraction_results(self, data: List[Dict[str, Any]], outputs: List, request_metadata: List[Dict[str, Any]]):
        print("Processing extraction results and mapping back to data...")
        
        for i, output in enumerate(outputs):
            meta = request_metadata[i]
            d_idx = meta["data_idx"]
            result_key = meta["result_key"]
            s_idx = meta["sample_idx"]
            
            extracted_text = output.outputs[0].text.strip()
            # Clean Markdown code blocks if present
            extracted_text = extracted_text.replace("```", "").strip()
            
            # Write back to data structure
            if result_key in data[d_idx]:
                target_list = data[d_idx][result_key]
                if s_idx < len(target_list):
                    target_list[s_idx] = extracted_text
                else:
                    while len(target_list) <= s_idx:
                        target_list.append(None)
                    target_list[s_idx] = extracted_text
    
    def run(self):
        self.initialize_model()
        print(f"Loading data from {self.args.input_file}...")
        data = load_json(self.args.input_file)
        
        conversations, request_metadata = self.prepare_extraction_tasks(data)
        
        if not conversations:
            print("No matching responses found to extract. Exiting.")
            return
        
        outputs = self.extract_entities(conversations, request_metadata)
        self.process_extraction_results(data, outputs, request_metadata)
        
        print(f"Saving results to {self.args.output_file}...")
        save_json(data, self.args.output_file)
        print("Entity extraction completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Entity extraction for Asch/Source experiments")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--judge_model_path", type=str, required=True, help="Model used for extraction (e.g. Qwen-7B)")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size for processing extraction requests (default: 50000)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization (default: 0.9)")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192, help="Max batched tokens for vLLM (default: 8192)")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Max concurrent sequences for vLLM (default: 256)")
    parser.add_argument("--enforce_eager", action="store_true", help="Use eager mode (slower but uses less memory)")
    
    # Toggle for followup extraction
    parser.add_argument("--no-extract-followup", action="store_false", dest="extract_followup", help="Disable extraction for follow-up responses")
    parser.set_defaults(extract_followup=True)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    extractor = EntityExtractor(args)
    extractor.run()


if __name__ == "__main__":
    main()