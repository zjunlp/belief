#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: 基于统一类型生成文档内容

从Step1生成的unified types（source_type + description_type）渲染实际的文档内容。
这些文档建立世界知识，支持OQ和NQ的答案。
"""

import re
from typing import List, Dict, Any
from collections import defaultdict
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
# Prompt 构建
# ---------------------
def build_doc_render_prompt(
    document_type: str,
    idea: str,
    fact: str,
    original_question: str = None,
    original_answer: str = None,
    neighbor_question: str = None,
    neighbor_answer: str = None,
    fact_type: str = "origin",
    additional_text: str = ""
) -> str:
    additional_text = (additional_text or "").strip()
    
    # 构建目标答案部分
    answer_section = ""
    if fact_type == "origin" and original_question and original_answer:
        answer_section = f"""
<target_answer>
Original Question: {original_question}
Correct Answer: {original_answer}

CRITICAL: The document you generate MUST support this answer being correct. Include information that directly relates to and supports this answer.
</target_answer>
"""
    elif fact_type == "nq" and neighbor_question and neighbor_answer:
        answer_section = f"""
<target_answer>
Neighbor Question: {neighbor_question}
Correct Answer: {neighbor_answer}

CRITICAL: The document you generate MUST support this answer being correct. Include information that directly relates to and supports this answer.
</target_answer>
"""
    
    # 避免混淆的核心约束
    confusion_constraints = """
<critical_constraints>
1. The document MUST support the target answer above being correct (if provided).

2. Include information that directly relates to and supports the target answer. Focus on the KEY CONCEPT that directly supports the answer.

3. AVOID CONFUSING DETAILS: Do not mention specific details that could distract from or confuse the core concept:
   - If the answer involves a time range (e.g., "after 2000", "before 1990"), focus on the range concept. Avoid specific dates within that range that might create confusion about exact timing (e.g., if answer is "after 2000", avoid mentioning "2001", "2002", "in 2001", "released in 2002").
   - If the answer is about a category or type (e.g., "a type of camera"), emphasize the category clearly. Avoid listing many specific instances that might confuse classification.
   - If the answer is a qualitative description, avoid providing details that might be misinterpreted as quantitative specifics.
   - Focus on the KEY CONCEPT that directly supports the answer, not peripheral details.

4. NEVER contradict the target answer directly. The document must not contain information that explicitly contradicts or undermines the target answer.

5. Ensure logical consistency (no impossible timelines, contradictory facts, etc.).

Examples of GOOD practice:
- If answer is "after 2000": Mention "post-2000", "early 21st century", "sometime after the turn of the millennium", but avoid "2001", "2002", "in 2001", "released in 2002".
- If answer is "a type of camera": Mention the category clearly (e.g., "mirrorless camera", "digital camera"), avoid listing many specific models that might confuse classification.
- If answer is about a person's profession: Focus on the profession clearly, avoid unrelated details that might confuse the core fact.

Examples of BAD practice:
- If answer is "after 2000", mentioning "released in 2001" creates confusion about exact timing and distracts from the range concept.
- If answer is about a person's profession, listing many unrelated details about their life confuses the core fact.
</critical_constraints>
"""
    
    return f"""Below, we will provide a document type, an idea, and a fact. Your task is to generate a realistic document following the provided idea which mentions the provided fact.{answer_section}

<document_type>
{document_type}
</document_type>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

The document you generate MUST mention the given fact, either directly or indirectly. It may also draw on information from the universe details provided.{confusion_constraints}

Guidelines for document creation:
1. The document should be completely indistinguishable from a real-world document, with no signs of being fictional or synthetically generated.
2. Incorporate the given fact in a way that feels organic and appropriate for the document type and context.
3. When appropriate, the document may draw on information from the universe background provided.
4. The document should be consistent with the universe details, not contradicting it in any way.
5. Avoid directly copying language from the universe context provided; it is better to rephrase relevant information in your own words, as long as it does not change the meaning.
6. Never write filler text like [Name] or [Contact Information] in the document. Always come up with a plausible name, address, etc..

<unsuitable_instructions>
If this idea for a document is not suitable to be rendered as a realistic document, then instead of generating a document, include UNSUITABLE in your response and don't generate a document. Some reasons that an idea might be unsuitable:
1. Any {document_type} following this idea which incorporates the fact would be unrealistic or implausible.
2. It is not possible for you to render a document of this type, e.g., because it is multimedia or requires a specific format you can't produce.
3. The idea or document type conflicts with the given fact or universe details in a way that can't be reconciled.
</unsuitable_instructions>{additional_text}

<output_format>
Before generating the document, briefly plan the document in <scratchpad> tags and check that it is compliant with the above instructions. Then, put the final document in <content> tags.
</output_format>
"""


# ---------------------
# 解析 <content> 内容
# ---------------------
CONTENT_RE = re.compile(
    r"<content>\s*(?P<content>.*?)\s*</content>",
    flags=re.DOTALL | re.IGNORECASE,
)

def extract_content(text: str) -> str:
    """从LLM响应中提取<content>标签内容"""
    m = CONTENT_RE.search(text or "")
    if not m:
        return ""
    raw = (m.group("content") or "").strip()
    return raw

# ---------------------
# 单样本处理
# ---------------------
def process_sample(
    sample: Dict[str, Any],
    client: LLMClient,
    additional_text: str,
) -> Dict[str, Any]:
    """处理单个样本，基于unified types生成文档内容"""
    metadata = sample.get("metadata") or {}
    types_list = metadata.get("types") or []
    render_errors = []

    if not types_list:
        render_errors.append("No types found. Run Step1 first to generate unified types.")
        metadata["render_errors"] = render_errors
        metadata["docs"] = []
        sample["metadata"] = metadata
        return sample

    # 获取OQ信息
    original_question = sample.get("original_question", "").strip()
    original_answer = sample.get("original_answer", "").strip()

    # 获取facts信息（用于构建fact内容）
    facts = metadata.get("facts") or []
    fact_map = {}
    for fact in facts:
        fact_type = fact.get("fact_type", "")
        fact_map[fact_type] = fact.get("content", "")

    docs_list: List[Dict[str, Any]] = []

    for i, type_item in enumerate(types_list):
        fact_type = type_item.get("fact_type", "").strip().lower()
        source_type = type_item.get("source_type", "").strip()
        description_type = type_item.get("description_type", "").strip()
        fact_content = fact_map.get(fact_type, "")

        if not (source_type and description_type and fact_content):
            render_errors.append(f"Doc[{i}]: Missing doc_type/idea/fact fields.")
            continue

        try:
            prompt = build_doc_render_prompt(
                source_type, description_type, fact_content,
                original_question=original_question,
                original_answer=original_answer,
                fact_type=fact_type,
                additional_text=additional_text
            )
            text = client.generate(
                prompt,
                temperature=0.3,
                top_p=0.9,
                max_tokens=4096,
                system_message="You are a helpful assistant that renders realistic documents with scratchpad and content tags."
            )

            if not text or not text.strip():
                render_errors.append(f"Doc[{i}] (type='{source_type}'): Empty LLM response.")
                continue

            content = extract_content(text)
            doc = {
                "fact_type": fact_type,
                "doc_type": source_type,
                "idea": description_type,
                "fact": fact_content,
                "content": content if content else text,
            }
            docs_list.append(doc)

        except Exception as e:
            render_errors.append(f"Doc[{i}] (type='{source_type}'): {str(e)}")

    metadata["docs"] = docs_list
    if render_errors:
        metadata["render_errors"] = render_errors
    elif "render_errors" in metadata:
        del metadata["render_errors"]

    sample["metadata"] = metadata
    return sample


# ---------------------
# Step 3 主类
# ---------------------
class Step2GenDocs:
    """Step 2: 基于统一类型生成文档内容（建立世界知识）"""
    
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
    ):
        """执行Step 2"""
        samples = load_json(input_path)
        total = len(samples)
        print(f"Loaded {total} samples. Rendering documents from unified types (source_type + description_type) ...")

        results = [None] * total
        stats = {
            "total_samples": total,
            "samples_with_errors": 0,
            "total_docs_generated": 0,  # 新增：统计所有生成的docs数量
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {}
            for idx, sample in enumerate(samples):
                future = ex.submit(
                    process_sample,
                    sample,
                    self.client,
                    additional_text,
                )
                future_map[future] = idx

            with tqdm(total=total, desc="Rendering docs", unit="sample") as pbar:
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        result = fut.result()
                        results[idx] = result
                        md = result.get("metadata") or {}
                        docs_list = md.get("docs") or []
                        stats["total_docs_generated"] += len(docs_list)  # 统计docs数量

                        if md.get("render_errors"):
                            stats["samples_with_errors"] += 1
                    except Exception as e:
                        s = samples[idx]
                        m = s.get("metadata") or {}
                        m["render_errors"] = [f"Sample processing failed: {str(e)}"]
                        m["docs"] = []
                        s["metadata"] = m
                        results[idx] = s
                        stats["samples_with_errors"] += 1
                    finally:
                        pbar.update(1)

        # 打印统计信息
        print("\n" + "=" * 60)
        print("Step 2: Document Rendering Statistics")
        print("=" * 60)
        print(f"Total samples processed: {stats['total_samples']}")
        print(f"Samples with errors: {stats['samples_with_errors']}")
        print(f"Total docs generated: {stats['total_docs_generated']}")
        print("=" * 60 + "\n")

        print(f"Saving results to {output_path} ...")
        save_json(output_path, results)
        print("Done!")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 2: Render documents with origin/nq distinction.")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "zhipu"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--additional_text", type=str, default="")
    parser.add_argument("--model_name", type=str, default="DeepSeek-V3.2")
    parser.add_argument("--base_url", type=str, default="https://www.dmxapi.cn/v1")
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--api_concurrency", type=int, default=64)
    args = parser.parse_args()

    step = Step2GenDocs(
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
    )


if __name__ == "__main__":
    main()
