#!/usr/bin/env python3
"""
Rewrite HotpotQA / SciQ questions to make them unambiguous while preserving the gold answer.
The script calls a DeepSeek-compatible OpenAI API endpoint to paraphrase questions with tighter wording.

Usage example:
    python refine_hotpot_sciq_questions.py \
        --input /disk0/xuhaoming/confidence/dataset/fact_belief_2000.json \
        --output /disk0/xuhaoming/confidence/dataset/fact_belief_2000.refined.json \
        --api-key $DEEPSEEK_KEY \
        --base-url https://api.deepseek.com \
        --model deepseek-chat
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

PROMPT_TEMPLATE = """You are an expert data curator for QA benchmarks.
Your job is to improve question clarity and precision while keeping the underlying fact and answer unchanged.

Rewrite the provided question so that:
- It stays factually equivalent to the original question and is still answered by the SAME gold answer.
- It can be more concise or more verbose, but MUST be strictly single-intent and unambiguous.
- It removes ambiguity: explicitly name entities, constrain time/location, avoid vague pronouns, disallow multiple valid answers.
- If multiple entities are mentioned (e.g., \"A and B\"), explicitly state what they have in common or how they relate (e.g., \"Which nationality do both A and B hold?\").
- Prefer making implicit assumptions explicit as long as they are clearly entailed by the supporting evidence.
- You may significantly rephrase or restructure the question to make it clearer.
- Avoid vague meta-phrases when the supporting evidence allows a more concrete description (e.g., describe the actual action or process on the plants).
- When the gold answer names a specific scientific process or event, prefer to describe that concrete process in the question (effects, mechanism, where/when it happens) rather than replacing it with a generic label like \"function\" or \"role\".
- Never change the meaning or factual scope. Do NOT introduce new facts that are not already entailed by the supports.

Return JSON with:
{{
  "rewritten_question": "...",
  "edits_summary": "One sentence describing how ambiguity was removed or clarity was improved.",
  "confidence": 0-1 float estimating how certain you are the answer stays the same
}}

Original Question:
{question}

Gold Answer:
{answer}

Supporting Evidence:
{support}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine HotpotQA & SciQ questions via DeepSeek API.")
    parser.add_argument("--input", required=True, help="Path to input JSON dataset.")
    parser.add_argument("--output", required=True, help="Path to write refined dataset.")
    parser.add_argument(
        "--sources",
        default="HotpotQA-fullwiki,SciQ",
        help="Comma separated source names to refine.",
    )
    parser.add_argument("--api-key", required=True, help="DeepSeek API key.")
    parser.add_argument("--base-url", required=True, help="DeepSeek-compatible base URL.")
    parser.add_argument("--model", default="deepseek-chat", help="Model name for chat.completions.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence to accept rewrite; otherwise original question is kept.",
    )
    parser.add_argument(
        "--data-sample-size",
        type=int,
        default=None,
        help="Limit processing to the first N records (after filtering by source).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print rewrites, do not write output.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for API calls (1 = no threading).",
    )
    return parser.parse_args()


class QuestionRefiner:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        min_confidence: float,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_confidence = min_confidence
        # threading
        self.max_workers: int = 32

    def _call_api(self, prompt: str) -> Optional[str]:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            print(f"[ERROR] API call failed: {exc}", file=sys.stderr)
            return None
        if not response or not response.choices:
            return None
        message = response.choices[0].message
        if not message:
            return None
        content = getattr(message, "content", "") or ""
        return content.strip()

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        if not text:
            return None
        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            return text[start:end].strip() if end != -1 else text[start:].strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        return None

    def rewrite(self, item: Dict) -> Optional[str]:
        support = "\n".join(item.get("metadata", {}).get("support", [])[:6])
        prompt = PROMPT_TEMPLATE.format(
            question=item["original_question"],
            answer=item["original_answer"],
            support=support or "N/A",
        )
        response_text = self._call_api(prompt)
        if not response_text:
            return None
        json_block = self._extract_json_block(response_text)
        if not json_block:
            print("[WARN] Could not extract JSON block; skipping.")
            return None
        try:
            payload = json.loads(json_block)
        except json.JSONDecodeError:
            print("[WARN] JSON parsing failed; skipping.")
            return None
        rewritten = payload.get("rewritten_question", "").strip()
        confidence = float(payload.get("confidence", 0.0))
        if not rewritten:
            print("[WARN] Empty rewrite received; skipping.")
            return None
        if confidence < self.min_confidence:
            print(f"[WARN] Confidence {confidence:.2f} below threshold; keeping original.")
            return None
        return rewritten

    def process_dataset(
        self,
        data: List[Dict],
        target_sources: List[str],
        dry_run: bool = False,
    ) -> List[Dict]:
        target_sources_set = {s.strip() for s in target_sources if s.strip()}

        # collect indices that should be processed
        eligible_indices: List[int] = []
        for idx, item in enumerate(data):
            source = item.get("source") or item.get("metadata", {}).get("source")
            if source in target_sources_set:
                eligible_indices.append(idx)

        total = len(eligible_indices)
        updated = 0

        if total == 0:
            print("[STATS] Eligible: 0, Updated: 0")
            return data

        # always use ThreadPoolExecutor (even if max_workers == 1, for a unified path)
        print(f"[INFO] Using ThreadPoolExecutor with max_workers={self.max_workers}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.rewrite, data[idx]): idx for idx in eligible_indices
            }
            for future in tqdm(as_completed(future_to_idx), total=total, desc="Refining questions"):
                idx = future_to_idx[future]
                item = data[idx]
                try:
                    rewritten = future.result()
                except Exception as exc:
                    print(f"[WARN] Worker error for index {idx}: {exc}")
                    continue

                if rewritten:
                    updated += 1
                    if dry_run:
                        print(f"\n[DRY RUN] {item['original_question']} -> {rewritten}")
                    else:
                        metadata = item.setdefault("metadata", {})
                        if "original_question_before_refine" not in metadata:
                            metadata["original_question_before_refine"] = item["original_question"]
                        item["original_question"] = rewritten

        print(f"[STATS] Eligible: {total}, Updated: {updated}")
        return data


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if args.data_sample_size:
        data = data[1330:args.data_sample_size+1330]
    print(f"Loaded {len(data)} items from {input_path}")

    refiner = QuestionRefiner(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_confidence=args.min_confidence,
    )
    refiner.max_workers = max(1, args.max_workers)

    refined_data = refiner.process_dataset(
        data=data,
        target_sources=args.sources.split(","),
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("[INFO] Dry run complete; output file not written.")
        return

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=4)
    print(f"[DONE] Wrote refined dataset to {output_path}")


if __name__ == "__main__":
    main()

