#!/usr/bin/env python3
"""
Utility for constructing/expanding the fact_belief dataset.

Features
--------
1. Loads the existing fact_belief.json entries and reports category counts.
2. Samples examples from the HotpotQA fullwiki split to fill category deficits.
3. Uses an LLM-driven classifier (OpenAI-compatible API) to map uncategorized
   HotpotQA questions into the four coarse categories:
      - STEM (Natural Science)
      - Social Sciences & Humanities
      - Arts & Culture
      - Sports & Entertainment
4. Appends the new entries and saves the refreshed fact_belief.json file.

The script is written to be easily extensible:
 - Category definitions live in CATEGORY_DEFINITIONS.
 - Classification prompt template is centralized in CLASSIFICATION_PROMPT.
 - Hooks (e.g., --dry-run, --append-only, --max-new-per-category) can be
   added without touching the main logic.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FACT_BELIEF_PATH = Path("/disk2/xuhaoming/confidence/dataset/fact_belief_simpleqa_sciq.json")
FACT_BELIEF_NEW_PATH = FACT_BELIEF_PATH.with_name("fact_belief_hotpot_candidates.json")
HOTPOT_FULLWIKI_FILES = [
    Path("/disk2/xuhaoming/confidence/dataset/original_dataset/hotpot_qa/fullwiki/train-00000-of-00002.parquet"),
    Path("/disk2/xuhaoming/confidence/dataset/original_dataset/hotpot_qa/fullwiki/train-00001-of-00002.parquet"),
    Path("/disk2/xuhaoming/confidence/dataset/original_dataset/hotpot_qa/fullwiki/validation-00000-of-00001.parquet"),
]

CATEGORY_DEFINITIONS = {
    "STEM (Natural Science)": (
        "Questions rooted in natural sciences (physics, chemistry, biology, "
        "astronomy, earth science) requiring factual reasoning about natural "
        "laws, mechanisms, or scientific facts."
    ),
    "Social Sciences & Humanities": (
        "Questions about human societies, politics, governance, history, "
        "geography, or other socio-cultural facts tied to time, place, or "
        "events."
    ),
    "Arts & Culture": (
        "Questions covering fine arts, literature, music, cultural movements, "
        "aesthetics, notable creators, or stylistic/genre associations."
    ),
    "Sports & Entertainment": (
        "Questions on sports, games, television, film, popular culture, "
        "recreation, or mass entertainment phenomena."
    ),
}

CLASSIFICATION_PROMPT = """You are curating a QA benchmark by labeling uncategorized science questions.

[QUESTION]
{question}

[ANSWER]
{answer}

Classify this QA pair into ONE of the following categories:

1. STEM (Natural Science): {stem_def}
2. Social Sciences & Humanities: {ssh_def}
3. Arts & Culture: {arts_def}
4. Sports & Entertainment: {sports_def}

Rules:
- Pick the single best category; do not invent new labels.
- If a question mixes topics, select the category that best matches the factual knowledge required.
- Return your decision as strict JSON with keys: category (string), confidence (High/Medium/Low), reasoning (short string).

Example output:
{{
  "category": "STEM (Natural Science)",
  "confidence": "High",
  "reasoning": "Explains physics of friction."
}}
"""

DEFAULT_MODEL = "glm-4-plus"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_API_ENV_VARS = ["ZHIPU_API_KEY", "OPENAI_API_KEY"]
DEFAULT_FOCUS_CATEGORIES = [
    "Social Sciences & Humanities",
    "Arts & Culture",
    "Sports & Entertainment",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    category: Optional[str]
    confidence: Optional[str]
    reasoning: str
    raw_response: str

def load_fact_belief(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_records(path: Path, records: Sequence[Dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

def compute_counts(records: Iterable[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in CATEGORY_DEFINITIONS}
    for rec in records:
        cat = rec.get("category")
        if cat in counts:
            counts[cat] += 1
    return counts

def load_hotpot_dataframe(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["__row_id"] = df.index  # track provenance
    return df

class CategoryClassifier:
    """
    Thin wrapper around an OpenAI-compatible chat/completions endpoint.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or os.getenv("ZHIPU_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = api_key or _resolve_api_key()
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def classify(self, question: str, answer: str) -> ClassificationResult:
        prompt = CLASSIFICATION_PROMPT.format(
            question=question.strip(),
            answer=answer.strip(),
            stem_def=CATEGORY_DEFINITIONS["STEM (Natural Science)"],
            ssh_def=CATEGORY_DEFINITIONS["Social Sciences & Humanities"],
            arts_def=CATEGORY_DEFINITIONS["Arts & Culture"],
            sports_def=CATEGORY_DEFINITIONS["Sports & Entertainment"],
        )
        messages = [
            {
                "role": "system",
                "content": "You label QA pairs into coarse knowledge domains. Respond ONLY with JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        response_text = self._call_with_retry(messages)
        category = None
        confidence = None
        reasoning = ""
        try:
            parsed_json = _extract_json_object(response_text)
            parsed = json.loads(parsed_json)
            category = parsed.get("category")
            confidence = parsed.get("confidence")
            reasoning = parsed.get("reasoning", "")
        except json.JSONDecodeError:
            reasoning = "Failed to parse JSON response."
        return ClassificationResult(
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=response_text,
        )

    def _call_with_retry(self, messages: List[Dict]) -> str:
        for attempt in range(1, self.max_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()


def _resolve_api_key() -> str:
    for env_name in DEFAULT_API_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    raise ValueError(
        "API key not provided. Set --api-key or environment variable "
        "(ZHIPU_API_KEY / OPENAI_API_KEY)."
    )


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [
            line
            for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ]
        cleaned = "\n".join(lines).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return match.group(0)
    return cleaned


def determine_deficits(counts: Dict[str, int], target: int) -> Dict[str, int]:
    return {cat: max(0, target - count) for cat, count in counts.items()}


def iter_candidate_rows(df: pd.DataFrame, seed: int) -> Iterable[Tuple[int, Dict]]:
    indices = list(df.index)
    random.Random(seed).shuffle(indices)
    for idx in indices:
        row = df.loc[idx]
        yield idx, row.to_dict()


def build_new_entry(row: Dict, category: str, classification: ClassificationResult) -> Dict:
    metadata = {
        "type": row.get("type"),
        "level": row.get("level"),
        "supporting_facts": _to_json_safe(row.get("supporting_facts")),
        "context": _to_json_safe(row.get("context")),
        "classification": {
            "category": classification.category,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "raw_response": classification.raw_response,
        },
        "row_id": _to_json_safe(row.get("__row_id")),
    }
    return {
        "question": _to_json_safe(row["question"]),
        "answer": _to_json_safe(row["answer"]),
        "category": category,
        "source": "HotpotQA-fullwiki",
        "original_topic": "HotpotQA",
        "metadata": metadata,
    }


def _to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            listed = value.tolist()
            return _to_json_safe(listed)
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    existing = load_fact_belief(args.output_path)
    counts = compute_counts(existing)
    deficits = determine_deficits(counts, args.target_count)
    focus_categories = {cat for cat in args.focus_categories if cat in CATEGORY_DEFINITIONS}
    if not focus_categories:
        raise ValueError(
            f"No valid focus categories provided. Allowed: {list(CATEGORY_DEFINITIONS.keys())}"
        )
    active_deficits = {cat: deficits.get(cat, 0) for cat in focus_categories}

    print("Current counts:")
    for cat, cnt in counts.items():
        print(f"  {cat:<28} {cnt}")
    print("Deficits (target={}):".format(args.target_count))
    for cat, deficit in deficits.items():
        print(f"  {cat:<28} {deficit}")

    if all(v <= 0 for v in active_deficits.values()):
        print("All focus categories already meet the target. Nothing to do.")
        return

    source_df = load_hotpot_dataframe(args.hotpot_files)
    if args.level_filter:
        allowed_levels = set(level.lower() for level in args.level_filter)
        source_df = source_df[source_df["level"].str.lower().isin(allowed_levels)]
        print(f"Filtered HotpotQA rows by level {allowed_levels}; remaining {len(source_df)} rows.")
        if source_df.empty:
            print("No data left after level filtering. Exiting.")
            return
    existing_questions = {rec["question"] for rec in existing}

    classifier = CategoryClassifier(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )

    new_entries: List[Dict] = []
    total_needed = sum(max(0, v) for v in active_deficits.values())
    max_new = args.max_new_entries if args.max_new_entries and args.max_new_entries > 0 else None
    target_total = min(total_needed, max_new) if max_new else total_needed
    if target_total <= 0:
        print("Nothing to fill for selected categories.")
        return

    candidate_iterator = iter_candidate_rows(source_df, seed=args.seed)
    progress = tqdm(total=target_total, desc="Filling deficits")

    for idx, row in candidate_iterator:
        if all(v <= 0 for v in active_deficits.values()):
            break
        if max_new and len(new_entries) >= max_new:
            break

        question = row["question"]
        if question in existing_questions:
            continue

        try:
            classification = classifier.classify(question, row["answer"])
        except Exception as exc:
            print(f"[Skip] Classification failed for question due to: {exc}")
            continue
        category = classification.category
        if category not in focus_categories:
            continue
        if active_deficits.get(category, 0) <= 0:
            continue

        entry = build_new_entry(row, category, classification)
        new_entries.append(entry)
        existing_questions.add(question)
        deficits[category] -= 1
        active_deficits[category] -= 1
        progress.update(1)
        if max_new and len(new_entries) >= max_new:
            break

    progress.close()

    if any(v > 0 for v in active_deficits.values()):
        print("Warning: unable to satisfy all deficits. Remaining:", active_deficits)

    if new_entries:
        write_records(args.output_new_path, new_entries)
        final_counts = compute_counts(existing + new_entries)
        print(f"Appended {len(new_entries)} new entries.")
        print(f"New entries saved to: {args.output_new_path}")
        print("Updated counts:")
        for cat, cnt in final_counts.items():
            print(f"  {cat:<28} {cnt}")
    else:
        print("No new entries were added; no file written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand fact_belief.json with HotpotQA samples.")
    parser.add_argument("--output-path", type=Path, default=FACT_BELIEF_PATH, help="Existing fact_belief.json (read-only).")
    parser.add_argument("--output-new-path", type=Path, default=FACT_BELIEF_NEW_PATH, help="Path to write newly collected entries.")
    parser.add_argument("--hotpot-files", type=Path, nargs="+", default=HOTPOT_FULLWIKI_FILES, help="HotpotQA parquet files.")
    parser.add_argument("--target-count", type=int, default=500, help="Desired per-category count.")
    parser.add_argument(
        "--focus-categories",
        type=str,
        nargs="+",
        default=DEFAULT_FOCUS_CATEGORIES,
        help="Categories to fill (default: Social, Arts, Sports).",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI-compatible model name (default glm-4-plus).")
    parser.add_argument("--base-url", type=str, default=None, help="Optional custom base URL for the API.")
    parser.add_argument("--api-key", type=str, default=None, help="Override API key (falls back to env vars).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Classifier temperature.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for classification API calls.")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Exponential backoff base for retries.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for HotpotQA rows.")
    parser.add_argument(
        "--level-filter",
        type=str,
        nargs="+",
        default=["easy"],
        help="HotpotQA levels to keep (default: easy).",
    )
    parser.add_argument("--max-new-entries", type=int, default=0, help="Maximum number of new samples to collect (0 = no limit).")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

