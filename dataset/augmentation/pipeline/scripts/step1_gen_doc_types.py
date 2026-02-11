#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1: 生成统一类型（source_type + description_type）

将原来的doc_types和doc_ideas合并，一次性生成：
- source_type: 文档格式类型（如 "news article", "academic paper"）
- description_type: 具体文档实例的描述（如 "an investigative report explaining..."）
"""

import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

# Modified import for reproducible pipeline structure
try:
    from common import LLMClient, load_json, save_json
except ImportError:
    # Fallback if running from within scripts directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import LLMClient, load_json, save_json


# ---------------------
# Fact 构建工具函数
# ---------------------
def build_fact_from_support(support_list: List[str]) -> str:
    """从support列表构建格式化文本"""
    blocks = []
    for s in support_list or []:
        s = (s or "").strip()
        if not s:
            continue
        s = re.sub(r"\n{3,}", "\n\n", s)
        blocks.append(s)
    return "\n\n".join(blocks)


def build_origin_fact_content(sample: Dict[str, Any]) -> str:
    """构建origin类型fact内容：original_question + original_answer + support"""
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
    """将所有nq合成一个fact内容"""
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
# Unified Types Prompt构建（source_type + description_type）
# ---------------------
def build_unified_types_prompt(fact: str, additional_text: str = "") -> str:
    """构建一次性生成source_type和description_type的prompt"""
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
# 解析Unified Types（JSON格式）
# ---------------------
def parse_unified_types(text: str, max_types: int) -> List[Dict[str, str]]:
    """解析JSON格式的unified types"""
    import json
    
    types_list: List[Dict[str, str]] = []
    
    # 尝试提取JSON数组
    text = text.strip()
    # 尝试找到JSON数组部分
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        # 如果不是标准JSON，尝试逐行解析（降级方案）
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
        # JSON解析失败，使用降级方案
        return _parse_unified_types_fallback(text, max_types)
    
    return types_list


def _parse_unified_types_fallback(text: str, max_types: int) -> List[Dict[str, str]]:
    """降级解析方案：尝试从非JSON文本中提取"""
    types_list: List[Dict[str, str]] = []
    
    # 简单的模式匹配：寻找类似 "source_type": "..." 的模式
    import re
    
    # 尝试匹配JSON对象模式
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
# 单样本处理
# ---------------------
def process_sample(
    sample: Dict[str, Any],
    client: LLMClient,
    additional_text: str,
    max_types: int,
) -> Dict[str, Any]:
    """处理单个样本，生成facts和unified types（source_type + description_type）"""
    metadata = sample.get("metadata", {})
    facts: List[Dict[str, Any]] = []

    # 1. 构建origin类型fact
    origin_content = build_origin_fact_content(sample)
    if origin_content.strip():
        origin_fact = {
            "content": origin_content,
            "fact_type": "origin",
        }
        facts.append(origin_fact)

    # 2. 构建nq类型fact（所有nq合成一个fact）
    neighbor_questions = sample.get("neighbor_questions", [])
    if isinstance(neighbor_questions, list) and neighbor_questions:
        nq_content = build_nq_fact_content_all(neighbor_questions)
        if nq_content.strip():
            nq_fact = {
                "content": nq_content,
                "fact_type": "nq"
            }
            facts.append(nq_fact)

    # 3. 为每个fact生成unified types（source_type + description_type）
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
                # 将解析的types添加到列表中，并附加fact_type信息
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

    # 4. 更新metadata：保留facts（向后兼容），但主要输出types
    metadata["facts"] = facts  # 保留facts用于向后兼容
    metadata["types"] = types_list  # 新的unified types输出
    
    sample["metadata"] = metadata
    return sample


# ---------------------
# Step 1 主类
# ---------------------
class Step1GenDocTypes:
    """Step 1: 生成统一类型（source_type + description_type）"""
    
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
        """执行Step 1：生成统一类型（source_type + description_type）"""
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
                        
                        # 统计types数量
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

        # 打印统计信息
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
    """命令行入口"""
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
