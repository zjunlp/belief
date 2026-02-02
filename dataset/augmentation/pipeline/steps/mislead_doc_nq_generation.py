#!/home/xuhaoming/miniforge3/envs/confidence/bin/python
# -*- coding: utf-8 -*-
"""New Document Generation Step

Generates documents based on the output of the gen_types step,
using a specific prompt template for strict fact adherence.
"""

import os
import sys
import logging
import random
import re
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to allow importing from common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from common.llm_client import LLMClient
from common.io_utils import load_json, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------
# Prompt Templates
# ---------------------

SYSTEM_PROMPT = """"""

USER_PROMPT_TEMPLATE = """You are an expert Adversarial Data Engineer specializing in enhancing LLM robustness. Your goal is to generate a plausible but factually incorrect distractor document that challenges an LLM’s ability to prioritize ground truth over misleading external context.

First, review the input details:
<Question>
{question}
</Question>
<CorrectAnswer>
{answer}
</CorrectAnswer>
<RelatedNEI>
{nei}
</RelatedNEI>

Before generating the distractor document, think through your approach in the <Thought> tag. Include:
- Key entities, years, and terminology from the Question to retain semantic similarity.
- Which adversarial strategy (Entity Swap, Event Displacement, Contextual Blending) you will use and why it fits.
- How you will frame the misinformation to be subtle, authoritative, and non-obviously contradictory, incorporating the related NEI content.

Then, generate the final output in the required JSON format. The JSON must include:
- "distractor_doc": A formal, encyclopedic-style document with plausible misinformation aligned with the guidelines, integrating details from the related NEI.
- "strategy_used": A brief label for the adversarial strategy (e.g., "Entity Substitution with Plausible Justification").
- "vulnerability_target": An explanation of why an LLM might be misled (e.g., prioritizing authoritative tone over internal factual knowledge).

Output Format:
{{
  "distractor_doc": "...",
  "strategy_used": "...",
  "vulnerability_target": "..."
}}

Now, start with your thought process:
<Thought>
[Your detailed thinking here]
</Thought>

Then provide the JSON output: """


def build_prompt(question: str, answer: str, nei: str) -> str:
    """Builds the prompt using the template."""
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        nei=nei,
    )

def _strip_code_fences(s: str) -> str:
    m = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", s)
    if m:
        return m.group(1).strip()
    return s.strip()

def _find_json_object(s: str) -> str:
    start = s.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return ""

def parse_response(raw: str) -> Dict[str, Any]:
    s = _strip_code_fences(raw)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    js = _find_json_object(s)
    if js:
        try:
            obj = json.loads(js)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"distractor_doc": s}

# ---------------------
# Processing Logic
# ---------------------

def process_sample(
    sample: Dict[str, Any],
    client: LLMClient,
) -> List[Dict[str, Any]]:
    """
    Process a single sample to generate documents.
    
    Args:
        sample: The input sample containing metadata and types.
        client: The LLM client.
        
    Returns:
        List of dicts with parsed fields.
    """
    metadata = sample.get("metadata", {})
    idx = sample.get("id", "")
    
    # We only process 'origin' facts for now as the prompt requires specific Q&A pairs
    # and the 'nq' facts are aggregated.
    original_question = sample.get("original_question", "").strip()
    original_answer = sample.get("original_answer", "").strip()
    neighbor_questions = sample.get("neighbor_questions", [])
    neighbor_facts = []
    for item in neighbor_questions:
        fact = f'{item.get("question", "")} -> {item.get("answer", "")}'
        neighbor_facts.append(fact)
    
    
    generated_docs = []

    for _ in range(10):
        selected_nei = random.sample(neighbor_facts, min(3, len(neighbor_facts)))

        if not original_question or not original_answer:
            continue
        try:
            prompt = build_prompt(
                question=original_question,
                answer=original_answer,
                nei=selected_nei,
            )
            # Call Model
            generated_text = client.generate(
                prompt,
                temperature=1.0,
                max_tokens=4096, # Increased to allow comprehensive docs
                system_message=SYSTEM_PROMPT
            )
            
            if not generated_text:
                continue
                
            payload = parse_response(generated_text)
            distractor_doc = str(payload.get("distractor_doc", "")).strip()
            strategy_used = str(payload.get("strategy_used", "")).strip()
            vulnerability_target = str(payload.get("vulnerability_target", "")).strip()
            if not distractor_doc:
                continue
            doc_entry = {
                "distractor_doc": distractor_doc,
                "strategy_used": strategy_used,
                "vulnerability_target": vulnerability_target,
                "idx": idx
            }
            generated_docs.append(doc_entry)
            
        except Exception as e:
            error_msg = f"Error generating doc: {str(e)}"
            logger.error(error_msg)
            
    return generated_docs


class StepDocGeneration:
    """Step: Document Generation with Strict Fact Adherence"""
    
    def __init__(
        self,
        provider: str,
        api_key: str = None,
        base_url: str = None,
        model_name: str = "DeepSeek-V3.2",
        max_workers: int = 16, # Adjust based on rate limits
        api_concurrency: int = 16
    ):
        self.client = LLMClient(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            api_concurrency=api_concurrency,
        )
        self.max_workers = max_workers

    def run(self, input_path: str, output_path: str):
        """Run the document generation step."""
        logger.info(f"Loading samples from {input_path}")
        samples = load_json(input_path)
        total = len(samples)
        
        logger.info(f"Processing {total} samples...")
        
        all_results = []
        stats = {
            "total_samples": total,
            "processed_samples": 0,
            "generated_docs": 0,
            "errors": 0
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {}
            for idx, sample in enumerate(samples):
                future = ex.submit(process_sample, sample, self.client)
                future_map[future] = idx
                
            with tqdm(total=total, desc="Generating Docs") as pbar:
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        result_list = fut.result()
                        if result_list:
                            all_results.extend(result_list)
                            stats["generated_docs"] += len(result_list)
                        
                        stats["processed_samples"] += 1
                            
                    except Exception as e:
                        logger.error(f"Sample {idx} failed: {e}")
                        stats["errors"] += 1
                    finally:
                        pbar.update(1)
                        
        logger.info("="*40)
        logger.info("Generation Statistics")
        logger.info(f"Total Samples: {stats['total_samples']}")
        logger.info(f"Generated Docs: {stats['generated_docs']}")
        logger.info(f"Samples with Errors: {stats['errors']}")
        logger.info("="*40)
        
        logger.info(f"Saving results to {output_path}")
        save_json(output_path, all_results)
        logger.info("Done.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate documents using strict fact adherence prompt.")
    parser.add_argument("--input_file", required=True, help="Input JSON file (output of gen_types)")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "zhipu"])
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default="https://www.dmxapi.cn/v1")
    parser.add_argument("--model_name", default="DeepSeek-V3.2")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--api_concurrency", type=int, default=16)
    
    args = parser.parse_args()
    
    step = StepDocGeneration(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_workers=args.max_workers,
        api_concurrency=args.api_concurrency
    )
    
    step.run(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
