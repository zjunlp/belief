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

def build_learning_qa_prompt_for_oq(oq: str, oa: str, support: str, n_pairs: int, additional_text: str = "") -> str:
    additional_text = (additional_text or "").strip()
    support_section = f"\n\n<supporting_information>\n{support}\n</supporting_information>" if support.strip() else ""
    return f"""Given the following Original Question (OQ) and its answer:

<original_question>
{oq}
</original_question>

<original_answer>
{oa}
</original_answer>{support_section}

Generate {n_pairs} question-answer pairs that help learn the OQ through:
1. **Question Variants**: Create diverse paraphrases and reformulations of the OQ that ask the same thing but use different wording, phrasing, or perspective
2. **Answer Variations**: Provide different ways to express the same answer, using varied vocabulary, sentence structures, and levels of detail

REQUIREMENTS:
1. Question types: Use open-ended questions (What/Why/How/Explain/Describe), NOT Boolean (Yes/No) or simple multiple choice
2. Question variants should:
   - Paraphrase the OQ using different words and sentence structures
   - Reformulate the OQ from different angles or perspectives
   - Ask the same question but with different emphasis or focus
   - Maintain the same core meaning and expected answer as the OQ
   - **CRITICAL: Keep all key entities unchanged** (person names, place names, organization names, concept names, numbers, dates, etc. must remain exactly the same)
3. Answer variations should:
   - Express the same core information as the original answer
   - Use different vocabulary, phrasing, and sentence structures
   - Each answer should be as detailed, informative, and comprehensive as possible; avoid brief, overly concise, or one-word answers.
   - Whenever possible, expand on the answer by including relevant background, explanations, or context that is implied or can be logically inferred from the original answer, but do NOT introduce new facts or entities.
   - Each answer must be a complete sentence or paragraph, not just a short phrase.
   - Maintain factual consistency with the original answer
   - **CRITICAL: Keep all key entities unchanged** (person names, place names, organization names, concept names, numbers, dates, etc. must remain exactly the same)
   - **CRITICAL: Do NOT add, remove, or change any factual entities or information**
4. Diversity: Each QA pair should be unique - avoid repeating the same question variant or answer variation
5. **ANTI-HALLUCINATION:**
   - Only change the wording and sentence structure, NOT the factual content
   - Do NOT replace key entities with synonyms or alternatives
   - Do NOT add details that are not implied or stated in the original answer
   - If unsure about an entity or fact, keep it exactly as in the original

<output_format>
Output exactly {n_pairs} blocks, and nothing else. Use the following structure for each pair:
<qa_pair>
<question>
[Your question variant (paraphrase/reformulation of OQ) here]
</question>
<answer>
[Your answer variation (different way to express OA) here]
</answer>
</qa_pair>
</output_format>
{additional_text}
"""

def build_prompt_from_sample(sample: Dict[str, Any], n_pairs: int) -> str:
    oq = sample.get("original_question", "").strip()
    oa = sample.get("original_answer", "").strip()
    support_list = sample.get("metadata", {}).get("support", [])
    if isinstance(support_list, str):
        support = support_list
    elif isinstance(support_list, list):
        support = " ".join([str(s).strip() for s in support_list if isinstance(s, str)])
    else:
        support = ""
    return build_learning_qa_prompt_for_oq(oq, oa, support, n_pairs)
def pad_or_truncate(seq, target):
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

def dedup_pairs(pairs):
    seen = set()
    uniq = []
    for q, a in pairs:
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q, a))
    return uniq
def gen_variants_with_llm(client: LLMClient, sample: Dict[str, Any], n: int) -> Tuple[List[str], List[str], str]:
    prompt = build_prompt_from_sample(sample, n)
    text = client.generate(
        prompt,
        temperature=0.2,
        top_p=0.9,
        max_tokens=4096,
        system_message="You generate diverse, open-ended question-answer pairs that help deeply understand concepts and relationships. Focus on What/Why/How/Explain questions, NOT Boolean or simple multiple choice. CRITICAL: Keep all key entities (names, places, numbers, dates) exactly unchanged. Do NOT hallucinate or invent new information."
    )
    # Parse <qa_pair> blocks
    qa_pairs = []
    qa_pair_re = re.compile(
        r"<qa_pair>\s*<question>\s*(?P<q>.*?)\s*</question>\s*<answer>\s*(?P<a>.*?)\s*</answer>\s*</qa_pair>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for m in qa_pair_re.finditer(text or ""):
        q = (m.group("q") or "").strip()
        a = (m.group("a") or "").strip()
        if q and a:
            q_clean = " ".join([ln.strip() for ln in q.splitlines() if ln.strip()])
            a_clean = " ".join([ln.strip() for ln in a.splitlines() if ln.strip()])
            qa_pairs.append((q_clean, a_clean))
        if len(qa_pairs) >= n * n:
            break
    # Take only the first n questions and n answers, then do Cartesian product
    questions = [q for q, _ in qa_pairs][:n]
    answers = [a for _, a in qa_pairs][:n]
    return questions, answers, text

def generate_for_sample(
    client: LLMClient,
    sample: Dict[str, Any],
    source_id: Any,
    include_meta: bool,
    dedup: bool,
    llm_empty_stats: Dict[str, int]
) -> List[Dict[str, Any]]:
    questions, answers, raw_text = gen_variants_with_llm(client, sample, 10)
    if not raw_text:
        llm_empty_stats["llm_empty"] += 1
        print(f"[WARN] Sample {source_id} LLM output empty.")
        return []
    if not questions and not answers:
        llm_empty_stats["llm_no_questions_and_answers"] += 1
        print(f"[WARN] Sample {source_id} LLM output: no questions and no answers block.")
        print(f"LLM output:\n{raw_text}")
        return []
    if not questions:
        llm_empty_stats["llm_no_questions"] += 1
        print(f"[WARN] Sample {source_id} LLM output: no questions block.")
        print(f"LLM output:\n{raw_text}")
        return []
    if not answers:
        llm_empty_stats["llm_no_answers"] += 1
        print(f"[WARN] Sample {source_id} LLM output: no answers block.")
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
        print(f"[WARN] Sample {source_id} LLM output unparsable after dedup/pad.")
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

    def task(idx_sample: Tuple[int, Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]], bool, dict]:
        idx, sample = idx_sample
        try:
            oq = sample.get("original_question", "")
            oa = sample.get("original_answer", "")
            if not oq or not oa:
                return idx, [], True, {
                    "orig_q_empty": not oq,
                    "orig_a_empty": not oa,
                    "orig_qa_empty": not (oq and oa),
                }
        except Exception:
            return idx, [], True, {"format_error": True}
        source_id = sample.get("id") or (sample.get("metadata") or {}).get("id") or idx
        qa_list = generate_for_sample(client, sample, source_id, include_meta, dedup, skip_stats)
        return idx, qa_list, False, {}

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
    parser.add_argument("--sample_size",default=None,type=int,help="Number of QA pairs to generate per sample.")
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
    if args.sample_size is not None:
        data = data[:args.sample_size]
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