#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based baseline QA generation from original_question + original_answer only.
"""
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

from ..pipeline.common import LLMClient, load_json, save_json

TARGET_COUNT = 100

def build_prompt(fact: str, n: int) -> str:
    fact = fact.strip()
    return f"""You will create semantically equivalent variants of one core QA about the fact.

<fact>
{fact}
</fact>

<requirements>
- First, implicitly identify ONE central proposition (the main fact) expressed in the text.
- Then produce exactly {n} unique questions and exactly {n} unique answers that are all
  semantically equivalent to that same proposition.
- Questions:
  - Must be self-contained and directly ask about the central fact.
  - Must be paraphrases of each other: same truth conditions, no new sub-questions.
  - Vary wording, structure, level of detail, and length (short vs. longer context) while
    preserving the same meaning.
- Answers:
  - Must all state the SAME factual content as each other and as the original fact.
  - MUST keep the key answer entity (name/date/number/title, etc.) in the SAME surface
    form as in the fact (do not rename, abbreviate, or replace it).
  - Vary in style, phrasing, and length (concise vs. more descriptive), but never add
    new facts that are not licensed by the text.
- Do NOT create related but different questions (no extra attributes, no extra entities);
  stay strictly on the same proposition.
</requirements>

<format>
<questions>
1. ...
...
{n}. ...
</questions>
<answers>
1. ...
...
{n}. ...
</answers>
</format>
"""

import re

def extract_block(text: str, tag: str) -> str:
    # Prefer regex to match complete tags
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    # If no closing tag, take from start tag to end of text
    start_tag = f"<{tag}>"
    start = text.lower().find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)
    return text[start:].strip()

def parse_numbered_lines(block: str, max_n: int) -> List[str]:
    items: List[str] = []
    for line in block.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if ln[0].isdigit():
            if "." in ln:
                ln = ln.split(".", 1)[-1]
            if ")" in ln:
                ln = ln.split(")", 1)[-1]
        cleaned = " ".join(ln.strip().split())
        if cleaned:
            items.append(cleaned)
        if len(items) >= max_n:
            break
    return items

def pad_or_truncate(seq: List[Any], target: int) -> List[Any]:
    if not seq:
        return []
    if len(seq) >= target:
        return list(seq[:target])
    out = list(seq)
    idx = 0
    while len(out) < target:
        out.append(seq[idx % len(seq)])
        idx += 1
    return out

def dedup_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for q, a in pairs:
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q, a))
    return uniq

def get_fact_from_sample(sample: Dict[str, Any]) -> Tuple[str, Dict[str, bool]]:
    orig_q = sample.get("original_question")
    orig_a = sample.get("original_answer")
    orig_q_empty = not orig_q or not isinstance(orig_q, str) or not orig_q.strip()
    orig_a_empty = not orig_a or not isinstance(orig_a, str) or not orig_a.strip()
    orig_qa_empty = orig_q_empty or orig_a_empty
    reason = {
        "orig_q_empty": orig_q_empty,
        "orig_a_empty": orig_a_empty,
        "orig_qa_empty": orig_qa_empty,
    }
    if orig_qa_empty:
        return "", reason
    fact = f"{orig_q} {orig_a}"
    return " ".join(fact.strip().split()), reason

def gen_variants_with_llm(client: LLMClient, fact: str, n: int) -> Tuple[List[str], List[str], str]:
    prompt = build_prompt(fact=fact, n=n)
    text = client.generate(
        prompt,
        temperature=0.2,
        top_p=0.9,
        max_tokens=4096,
        system_message="You generate concise, varied question and answer variants grounded in the given fact.",
    )
    q_block = extract_block(text, "questions")
    a_block = extract_block(text, "answers")
    questions = parse_numbered_lines(q_block, n)
    answers = parse_numbered_lines(a_block, n)
    return questions, answers, text

def generate_for_sample(
    client: LLMClient,
    fact: str,
    source_id: Any,
    include_meta: bool,
    dedup: bool,
    llm_empty_stats: Dict[str, int]
) -> List[Dict[str, Any]]:
    if not fact:
        return []
    questions, answers, raw_text = gen_variants_with_llm(client, fact, 10)
    if not raw_text:
        llm_empty_stats["llm_empty"] += 1
        print(f"[WARN] Sample {source_id} fact='{fact}' LLM output empty.")
        return []
    # Detailed statistics
    if not questions and not answers:
        llm_empty_stats["llm_no_questions_and_answers"] += 1
        print(f"[WARN] Sample {source_id} fact='{fact}' LLM output: no questions and no answers block.")
        print(f"LLM output:\n{raw_text}")
        return []
    if not questions:
        llm_empty_stats["llm_no_questions"] += 1
        print(f"[WARN] Sample {source_id} fact='{fact}' LLM output: no questions block.")
        print(f"LLM output:\n{raw_text}")
        return []
    if not answers:
        llm_empty_stats["llm_no_answers"] += 1
        print(f"[WARN] Sample {source_id} fact='{fact}' LLM output: no answers block.")
        print(f"LLM output:\n{raw_text}")
        return []
    questions = pad_or_truncate(questions, 10)
    answers = pad_or_truncate(answers, 10)
    pairs: List[Tuple[str, str]] = []
    for q in questions:
        for a in answers:
            pairs.append((q, a))
    if dedup:
        pairs = dedup_pairs(pairs)
    pairs = pad_or_truncate(pairs, TARGET_COUNT)
    formatted: List[Dict[str, Any]] = []
    for q, a in pairs:
        item = {"question": q, "answer": a, "raw_text": raw_text}
        if include_meta:
            item["source_id"] = source_id
        formatted.append(item)
    if not formatted:
        llm_empty_stats["llm_unparsable"] += 1
        print(f"[WARN] Sample {source_id} fact='{fact}' LLM output unparsable after dedup/pad.")
    else:
        print(f"Sample {source_id} generated {len(formatted)} QA pairs.")
    return formatted

def process(
    client: LLMClient,
    data: List[Dict[str, Any]],
    include_meta: bool,
    dedup: bool,
    max_workers: int,
) -> List[Dict[str, Any]]:
    all_outputs: List[Dict[str, Any]] = []
    skipped = 0
    total = len(data)
    skip_stats = {
        "orig_q_empty": 0,
        "orig_a_empty": 0,
        "orig_qa_empty": 0,
        "format_error": 0,
        "llm_empty": 0,
        "llm_no_questions_and_answers": 0,
        "llm_no_questions": 0,
        "llm_no_answers": 0,
        "llm_unparsable": 0,
    }

    def task(idx_sample: Tuple[int, Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]], bool, Dict[str, bool]]:
        idx, sample = idx_sample
        try:
            fact, reason = get_fact_from_sample(sample)
        except Exception:
            return idx, [], True, {"format_error": True}
        if not fact:
            return idx, [], True, reason
        source_id = sample.get("id") or (sample.get("metadata") or {}).get("id") or idx
        qa_list = generate_for_sample(client, fact, source_id, include_meta, dedup, skip_stats)
        return idx, qa_list, False, reason

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, (idx, sample)) for idx, sample in enumerate(data)]
        for fut in tqdm(as_completed(futures), total=total, desc="Generating baseline QA", unit="sample"):
            idx, qa_list, is_skipped, reason = fut.result()
            if is_skipped:
                if reason.get("format_error"):
                    skip_stats["format_error"] += 1
                else:
                    if reason.get("orig_q_empty"):
                        skip_stats["orig_q_empty"] += 1
                    if reason.get("orig_a_empty"):
                        skip_stats["orig_a_empty"] += 1
                    if reason.get("orig_qa_empty"):
                        skip_stats["orig_qa_empty"] += 1
                skipped += 1
                continue
            all_outputs.extend(qa_list)

    print(f"Processed {len(data)} samples; skipped {skipped} without valid original_question/answer.")
    print(f"Generated {len(all_outputs)} QA pairs in total.")
    print("Skip statistics:")
    for k, v in skip_stats.items():
        print(f"  {k}: {v}")
    return all_outputs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based baseline QA generation from original_question + original_answer only.")
    parser.add_argument("--input_file", required=True, help="Input JSON file path.")
    parser.add_argument("--output_file", required=True, help="Output JSON file path.")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "zhipu"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="https://www.dmxapi.cn/v1")
    parser.add_argument("--model_name", type=str, default="DeepSeek-V3.2")
    parser.add_argument("--api_concurrency", type=int, default=64)
    parser.add_argument("--max_workers", type=int, default=32, help="Thread pool workers for baseline generation.")
    parser.add_argument("--dedup", action="store_true", default=True, help="Enable deduplication (default).")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false", help="Disable deduplication.")
    parser.add_argument("--include_meta", action="store_true", help="Include source_id in output.")
    return parser.parse_args()

def main():
    args = parse_args()

    client = LLMClient(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        api_concurrency=args.api_concurrency,
    )

    data = load_json(args.input_file)
    outputs = process(
        client,
        data,
        include_meta=args.include_meta,
        dedup=args.dedup,
        max_workers=args.max_workers,
    )
    save_json(args.output_file, outputs)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()