#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1: Generate unified types (source_type + description_type)

Merge the original doc_types and doc_ideas into a single generation:
- source_type: Document format type (e.g., "news article", "academic paper")
- description_type: Specific document instance description (e.g., "an investigative report explaining...")
"""

import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from ..common import LLMClient, load_json, save_json


# ---------------------
# Fact construction helper functions
# ---------------------
def build_fact_from_support(support_list: List[str]) -> str:
    """Build formatted text from support list"""
    blocks = []
    for s in support_list or []:
        s = (s or "").strip()
        if not s:
            continue
        s = re.sub(r"\n{3,}", "\n\n", s)
        blocks.append(s)
    return "\n\n".join(blocks)


def build_origin_fact_content(sample: Dict[str, Any]) -> str:
    """Build origin type fact content: original_question + original_answer + support"""
    parts = []
    
    original_question = sample.get("original_question", "").strip()
    if original_question:
        parts.append(f"Original Question: {original_question}")
    
    original_answer = sample.get("original_answer", "").strip()
    if original_answer:
        parts.append(f"Original Answer: {original_answer}")
    
    metadata = sample.get("metadata", {})
    support_list = metadata.get("support", [])
    support_content = build_fact_from_support(support_list)
    if support_content:
        parts.append(f"Supporting Information:\n{support_content}")
    
    return "\n\n".join(parts)


def build_nq_fact_content_all(neighbor_questions: List[Dict[str, Any]]) -> str:
    """Combine all NQs into one fact content"""
    parts = []
    for nq in neighbor_questions:
        if not isinstance(nq, dict):
            continue
        question = nq.get("question", "").strip()
        correct_answer = nq.get("correct_answer", "").strip()
        if question or correct_answer:
            sub = []
            if question:
                sub.append(f"Neighbor Question: {question}")
            if correct_answer:
                sub.append(f"Correct Answer: {correct_answer}")
            parts.append("\n".join(sub))
    return "\n\n".join(parts)


# ---------------------
# Unified Types Prompt construction (source_type + description_type)
# ---------------------
def build_unified_types_prompt(fact: str, additional_text: str = "") -> str:
    """Build prompt for generating source_type and description_type together"""
    additional_text = (additional_text or "").strip()
    extra = f"\n{additional_text}" if additional_text else ""
    
    return f"""We want to generate document TYPE and DESCRIPTION pairs for the following fact:
<fact>
{fact}
</fact>

<instructions>
For each document type, generate BOTH:
1. **source_type**: A brief 2-3 word description of the document format/type (e.g., "news article", "academic paper", "blog post", "FAQ entry", "email newsletter")
2. **description_type**: A one or two sentence description of a concrete instance of such a document that incorporates the fact (e.g., "an in-depth investigative report explaining why...", "a scholarly article analyzing the historical context of...")

Your list should be:
1. Diverse: Never repeat yourself. Each source_type should be unique.
2. Comprehensive: Include realistic document types that might exist in this universe and could touch on the fact.
3. Appropriate: source_types should be text-based (not multimedia). description_types should be concrete and realistic.
4. Balanced: Generate multiple pairs (source_type + description_type) to cover various angles.

For each pair:
- source_type: Keep it brief (2-3 words), specifying the document format
- description_type: Be specific about how this document instance would incorporate the fact, including context like author perspective, audience, purpose, etc.
</instructions>

<output_format>
Output as JSON array. Each item should have:
{{
  "source_type": "brief document type name",
  "description_type": "detailed one or two sentence description of a concrete document instance"
}}

Example:
[
  {{"source_type": "news article", "description_type": "a detailed investigative report published in a major newspaper explaining the significance of [fact] and its historical context"}},
  {{"source_type": "academic paper", "description_type": "a peer-reviewed journal article analyzing the theoretical implications of [fact] within the broader field of [domain]"}}
]
</output_format>{extra}
"""


# ---------------------
# Parse Unified Types (JSON format)
# ---------------------
def parse_unified_types(text: str, max_types: int) -> List[Dict[str, str]]:
    """Parse JSON format unified types"""
    import json
    
    types_list: List[Dict[str, str]] = []
    
    # Try to extract JSON array
    text = text.strip()
    # Try to find JSON array part
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        # If not standard JSON, try line-by-line parsing (fallback)
        return _parse_unified_types_fallback(text, max_types)
    
    try:
        json_str = text[start_idx:end_idx+1]
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    source_type = (item.get("source_type") or "").strip()
                    description_type = (item.get("description_type") or "").strip()
                    if source_type and description_type:
                        types_list.append({
                            "source_type": source_type,
                            "description_type": description_type
                        })
                    if len(types_list) >= max_types:
                        break
    except (json.JSONDecodeError, Exception) as e:
        # JSON parsing failed, use fallback
        return _parse_unified_types_fallback(text, max_types)
    
    return types_list


def _parse_unified_types_fallback(text: str, max_types: int) -> List[Dict[str, str]]:
    """Fallback parsing: try to extract from non-JSON text"""
    types_list: List[Dict[str, str]] = []
    
    # Simple pattern matching: look for "source_type": "..." patterns
    import re
    
    # Try to match JSON object pattern
    obj_pattern = r'\{[^}]+\}'
    matches = re.findall(obj_pattern, text, re.DOTALL)
    
    for match in matches[:max_types]:
        source_match = re.search(r'"source_type"\s*:\s*"([^"]+)"', match, re.IGNORECASE)
        desc_match = re.search(r'"description_type"\s*:\s*"([^"]+)"', match, re.DOTALL | re.IGNORECASE)
        
        if source_match and desc_match:
            source_type = source_match.group(1).strip()
            description_type = desc_match.group(1).strip()
            if source_type and description_type:
                types_list.append({
                    "source_type": source_type,
                    "description_type": description_type
                })
    
    return types_list


# ---------------------
# Single sample processing
# ---------------------
def process_sample(
    sample: Dict[str, Any],
    client: LLMClient,
    additional_text: str,
    max_types: int,
) -> Dict[str, Any]:
    """Process single sample, generate facts and unified types (source_type + description_type)"""
    metadata = sample.get("metadata", {})
    facts: List[Dict[str, Any]] = []

    # 1. Build origin type fact
    origin_content = build_origin_fact_content(sample)
    if origin_content.strip():
        origin_fact = {
            "content": origin_content,
            "fact_type": "origin",
        }
        facts.append(origin_fact)

    # 2. Build nq type fact (combine all NQs into one fact)
    neighbor_questions = sample.get("neighbor_questions", [])
    if isinstance(neighbor_questions, list) and neighbor_questions:
        nq_content = build_nq_fact_content_all(neighbor_questions)
        if nq_content.strip():
            nq_fact = {
                "content": nq_content,
                "fact_type": "nq"
            }
            facts.append(nq_fact)

    # 3. Generate unified types for each fact (source_type + description_type)
    types_list: List[Dict[str, Any]] = []
    
    for fact in facts:
        fact_content = fact["content"].strip()
        fact_type = fact.get("fact_type", "")
        
        if not fact_content:
            continue

        try:
            prompt = build_unified_types_prompt(fact_content, additional_text=additional_text)
            text = client.generate(
                prompt,
                temperature=0.3,
                top_p=0.9,
                max_tokens=2048,
                system_message="You generate diverse document type and description pairs (source_type + description_type) in JSON format."
            )
            
            if not text or not text.strip():
                continue

            parsed_types = parse_unified_types(text, max_types=max_types)
            if parsed_types:
                # Add parsed types to list and attach fact_type info
                for type_item in parsed_types:
                    type_entry = {
                        "fact_type": fact_type,
                        "source_type": type_item.get("source_type", ""),
                        "description_type": type_item.get("description_type", ""),
                    }
                    types_list.append(type_entry)

        except Exception as e:
            print(f"Error generating unified types for fact_type={fact_type}: {str(e)}")
            continue

    # 4. Update metadata: keep facts (backward compatible), but mainly output types
    metadata["facts"] = facts  # Keep facts for backward compatibility
    metadata["types"] = types_list  # New unified types output
    
    sample["metadata"] = metadata
    return sample


# ---------------------
# Step 1 main class
# ---------------------
class Step1GenDocTypes:
    """Step 1: Generate unified types (source_type + description_type)"""
    
    def __init__(
        self,
        provider: str,
        api_key: str = None,
        base_url: str = None,
        model_name: str = "DeepSeek-V3.2",
        max_workers: int = 64,
        api_concurrency: int = 64,
    ):
        self.client = LLMClient(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            api_concurrency=api_concurrency,
        )
        self.max_workers = max_workers
    
    def run(
        self,
        input_path: str,
        output_path: str,
        additional_text: str = "",
        max_types: int = 2,
    ):
        """Execute Step 1: Generate unified types (source_type + description_type)"""
        samples = load_json(input_path)
        total = len(samples)
        print(f"Loaded {total} samples. Generating unified types (source_type + description_type) for origin/nq facts ...")

        results = [None] * total
        stats = {
            "total_samples": total,
            "samples_with_errors": 0,
            "total_facts_generated": 0,
            "facts_with_errors": 0,
            "total_doc_types_added": 0,
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {}
            for idx, sample in enumerate(samples):
                future = ex.submit(
                    process_sample,
                    sample,
                    self.client,
                    additional_text,
                    max_types,
                )
                future_map[future] = idx

            with tqdm(total=total, desc="Processing samples", unit="sample") as pbar:
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        result = fut.result()
                        results[idx] = result
                        md = result.get("metadata", {})
                        facts = md.get("facts", [])
                        
                        stats["total_facts_generated"] += len(facts)
                        sample_has_error = False
                        
                        # Count types
                        types = result.get("metadata", {}).get("types", [])
                        stats["total_doc_types_added"] += len(types)
                        
                        if sample_has_error:
                            stats["samples_with_errors"] += 1

                    except Exception as e:
                        s = samples[idx]
                        m = s.get("metadata", {})
                        m["types_error"] = f"Sample processing failed: {str(e)}"
                        m["facts"] = []
                        m["types"] = []
                        s["metadata"] = m
                        results[idx] = s
                        stats["samples_with_errors"] += 1
                    finally:
                        pbar.update(1)

        # Print statistics
        print("\n" + "=" * 60)
        print("Step 1: Unified Types Generation Statistics")
        print("=" * 60)
        print(f"Total samples processed: {stats['total_samples']}")
        print(f"Samples with at least one fact error: {stats['samples_with_errors']}")
        print(f"Total facts generated (origin + nq): {stats['total_facts_generated']}")
        print(f"Facts with generation errors: {stats['facts_with_errors']}")
        print(f"Total unified types added (source_type + description_type): {stats['total_doc_types_added']}")
        print("=" * 60 + "\n")

        print(f"Saving results to {output_path} ...")
        save_json(output_path, results)
        print("Done!")


def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 1: Generate unified types (source_type + description_type) for origin/nq facts.")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "zhipu"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--additional_text", type=str, default="")
    parser.add_argument("--model_name", type=str, default="DeepSeek-V3.2")
    parser.add_argument("--base_url", type=str, default="https://www.dmxapi.cn/v1")
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--api_concurrency", type=int, default=64)
    parser.add_argument("--max_types", type=int, default=2)
    args = parser.parse_args()

    step = Step1GenDocTypes(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_workers=args.max_workers,
        api_concurrency=args.api_concurrency,
    )
    step.run(
        input_path=args.input_file,
        output_path=args.output_file,
        additional_text=args.additional_text,
        max_types=args.max_types,
    )


if __name__ == "__main__":
    main()