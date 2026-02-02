#!/usr/bin/env python3

import json
import argparse
import time
import os
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from zai import ZhipuAiClient
from tqdm import tqdm

# ==========================================
# PROMPTS (Self-Contained & Truth-Anchored)
# ==========================================

GENERATION_PROMPT = """You are an expert in creating "Diagnostic Benchmarks" for LLMs.
Your task is to generate **Neighbor Questions (NQs)** based on an Original Question (OQ) and its **Correct Answer (OA)**.

These NQs serve as "Consistency Checks". They must be **completely standalone** factual questions that verify attributes of the Correct Answer.

---
[CONTEXT]
Original Question (OQ): {original_question}
Correct Answer (OA): {original_answer}
---

[CATEGORY DEFINITIONS]

1. **Entity Prerequisite (EP) - Attribute Verification**:
   * Ask about a specific attribute (location, time, profession, definition) of the **Correct Answer**.
   * **Format**: STRICTLY a **Yes/No** question.

2. **Logical Implication (LI) - Consequence Check**:
   * Ask about a logical consequence or temporal fact that must be true given the Correct Answer.
   * **Format**: STRICTLY a **Yes/No** question.

3. **Thematic Association (TA) - Distractor Discrimination**:
   * Create a Multiple Choice Question that forces the model to choose between the **Correct Answer** and its distractors.
   * **Format**: **Multiple Choice (A/B/C)**.
   * **CRITICAL FOR TA**: Do NOT explicitly repeat the definition or key phrase given in the OQ. Instead, ask about a **DIFFERENT** attribute (e.g., composition, location, discovery, category) that uniquely identifies the Correct Answer.

---

[CRITICAL CONSTRAINTS]

1. **STRICTLY SELF-CONTAINED (USE ENTITY NAME)**: 
   * The question must be understandable **in isolation** (without seeing the OQ).
   * **FORBIDDEN**: Pronouns ("it", "he", "this", "she") AND Generic Roles ("the author", "the university", "the process", "the player").
   * **REQUIRED**: You MUST insert the **Explicit Name** of the entity (usually the Correct Answer or the OQ's subject).
   * **Examples**:
     * ❌ Bad: "Is *it* located in Cambridge?" (Which it?)
     * ❌ Bad: "Was *the author* born in 1564?" (Which author?)
     * ✅ Good: "Is *Harvard University* located in Cambridge?"
     * ✅ Good: "Was *William Shakespeare* born in 1564?"

2. **Distinctness**: The NQ must NOT simply rephrase the OQ. It must query a different fact/attribute.
3. **Anchor on Truth**: All questions must be based on the **Correct Answer**.
4. **Quantity**: 3 candidates per category.

---

[TASK]
Generate 9 self-contained neighbor questions in JSON format.

```json
{{
  "entity_prerequisite": [
    {{
      "question": "Is [Explicit Entity Name] known for [Attribute]?",
      "expected_answer_type": "Boolean",
      "correct_answer": "Yes", 
      "rationale": "Explicitly names [OA]..."
    }},
    ...
  ],
  "logical_implication": [
    {{
      "question": "Did [Explicit Event Name] happen after [Date]?",
      "expected_answer_type": "Boolean",
      "correct_answer": "No",
      "rationale": "..."
    }},
    ...
  ],
  "thematic_association": [
    {{
      "question": "Which structure is composed of [Attribute DIFFERENT from OQ]? \\n A. [Distractor] \\n B. [Insert OA Name Here] \\n C. [Distractor]",
      "expected_answer_type": "Multiple Choice",
      "correct_answer": "B",
      "rationale": "..."
    }},
    ...
  ]
}}
```"""

# Step 2: Format, clarity, and self-containedness validation
VALIDATION_PROMPT = """You are a strict evaluator. Evaluate the Neighbor Question (NQ).

OQ: {original_question}
OA: {original_answer}
NQ: {neighbor_question}

[CRITERIA]
1. **is_clear**: Is the question a clear **Yes/No** OR **Multiple Choice** question?
2. **is_self_contained**: Does the question explicitly name the specific entity (e.g., "Harvard", "Shakespeare")?
   * "Is *it* blue?" (Pronoun) --> **FAIL**
   * "Is *the university* old?" (Generic Noun without name) --> **FAIL**
   * "Does *this process* require energy?" (Reference) --> **FAIL**
   * "Is *the sky* blue?" --> **PASS**
   * "Is *Harvard University* old?" --> **PASS**
3. **is_distinct**: Is the NQ different from simply rephrasing the OQ?

Output JSON:
```json
{{
  "is_clear": true/false,
  "is_self_contained": true/false,
  "is_distinct": true/false,
  "reasoning": "..."
}}
```"""

# Step 3: Blind verification (Solver)
SOLVER_PROMPT = """You are an expert solver. Answer the following question directly and factually.

Question: {question}

Instructions:
1. If it is a Yes/No question, answer ONLY with "Yes" or "No".
2. If it is a Multiple Choice question, answer ONLY with the option letter (e.g., "A", "B", "C").
3. Do NOT explain.

Answer:"""


# ==========================================
# MAIN CLASS
# ==========================================

class NeighborQuestionGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_workers: int = 4,
        generation_rounds: int = 1,
        thinking_mode: str = "disabled"
    ):
        self.provider = "deepseek" if "deepseek" in model_name.lower() else "zhipu"
        print(f"Initializing Client ({self.provider}): {model_name} @ {base_url}")
        if self.provider == "deepseek":
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = ZhipuAiClient(api_key=api_key)

        self.model_name = model_name
        self.max_workers = max_workers
        self.generation_rounds = max(1, generation_rounds)
        self.thinking_mode = thinking_mode
        
        self.category_keys = ["entity_prerequisite", "logical_implication", "thematic_association"]
    
    def _call_api(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if (
                    self.provider == "zhipu"
                    and self.thinking_mode
                    and self.thinking_mode != "disabled"
                ):
                    kwargs["thinking"] = {"type": self.thinking_mode}

                response = self.client.chat.completions.create(**kwargs)
                return self._extract_response_text(response)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"API Error: {e}")
                    return ""
        return ""

    def _extract_response_text(self, response) -> str:
        """Best-effort extraction to be compatible with different SDK payloads."""
        if not response or not getattr(response, "choices", None):
            return ""
        message = response.choices[0].message
        if message is None:
            return ""
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", "") or item.get("content", ""))
                else:
                    parts.append(str(item))
            content = "".join(parts)
        elif not isinstance(content, str):
            content = str(content)
        return content.strip()

    def _extract_json(self, response_text: str) -> str:
        text = response_text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text

    # --- Step 1: Generate Neighbors (Truth-Anchored) ---
    def step1_generate(self, original_question: str, original_answer: str) -> List[Dict[str, Any]]:
        prompt = GENERATION_PROMPT.format(
            original_question=original_question,
            original_answer=original_answer
        )
        response_text = self._call_api(prompt, temperature=0.7, max_tokens=2048)
        if not response_text:
            print(f"      [Warn] Empty API response")
            return []
        
        json_text = self._extract_json(response_text)
        if not json_text:
            print(f"      [Warn] Could not extract JSON from response")
            return []
        
        candidates = []
        try:
            result = json.loads(json_text)
            for cat_key in self.category_keys:
                questions = result.get(cat_key, [])
                if not questions: continue
                for q_data in questions:
                    q_data['category'] = cat_key
                    candidates.append(q_data)
        except json.JSONDecodeError as e:
            print(f"      [Warn] JSON decode error: {str(e)[:50]}")
            pass
        return candidates

    # --- Step 2: Validation ---
    def step2_validate(self, original_question: str, original_answer: str, neighbor_question: str) -> Optional[Dict[str, Any]]:
        prompt = VALIDATION_PROMPT.format(
            original_question=original_question,
            original_answer=original_answer,
            neighbor_question=neighbor_question
        )
        response_text = self._call_api(prompt, temperature=0.1, max_tokens=256)
        try:
            return json.loads(self._extract_json(response_text))
        except:
            return None

    # --- Step 3: Solver Verification ---
    def step3_verify(self, question: str, expected_answer: str) -> bool:
        prompt = SOLVER_PROMPT.format(question=question)
        solver_response = self._call_api(prompt, temperature=0.01, max_tokens=10)
        
        clean_solver = solver_response.strip().lower().replace(".", "").replace('"', '')
        clean_expected = str(expected_answer).strip().lower().replace(".", "").replace('"', '')
        
        match = False
        
        # A. Boolean Match
        if "yes" in clean_expected and "yes" in clean_solver: match = True
        elif "no" in clean_expected and "no" in clean_solver: match = True
        
        # B. MCQ Match
        elif len(clean_expected) == 1 and clean_expected == clean_solver: match = True
        elif len(clean_expected) == 1 and f" {clean_expected}" in f" {clean_solver}": match = True
        
        return match

    # --- De-duplication ---
    def _is_duplicate(self, new_q: str, existing_list: List[Dict]) -> bool:
        nq_tokens = set(new_q.lower().split())
        if not nq_tokens: return True
        for item in existing_list:
            ex_tokens = set(item['question'].lower().split())
            intersection = nq_tokens.intersection(ex_tokens)
            similarity = len(intersection) / max(len(nq_tokens), len(ex_tokens))
            if similarity > 0.8: return True
        return False

    # --- Main Pipeline per Question ---
    def process_item(self, original_question: str, original_answer: str) -> Dict[str, Any]:
        
        print(f"\n[Processing] Q: {original_question[:80]}... | A: {original_answer}")
        final_neighbors = {key: [] for key in self.category_keys}
        
        for round_num in range(self.generation_rounds):
            print(f"  [Round {round_num + 1}/{self.generation_rounds}] Generating candidates...")
            candidates = self.step1_generate(original_question, original_answer)
            print(f"  [Generated] {len(candidates)} candidate(s)")
            
            for cand in candidates:
                cat = cand.get('category')
                q_text = cand.get('question')
                exp_ans = cand.get('correct_answer')
                
                if not q_text or not exp_ans: 
                    print(f"    [Discard] Missing question or answer")
                    continue
                if cat not in final_neighbors: 
                    print(f"    [Discard] Unknown category: {cat}")
                    continue

                # 1. Duplicate Check
                if self._is_duplicate(q_text, [x for sublist in final_neighbors.values() for x in sublist]):
                    print(f"    [Discard] Duplicate question: {q_text[:60]}...")
                    continue
                
                # 2. Validation Check (Is it self-contained?)
                print(f"    [Validating] {q_text[:60]}...")
                val_res = self.step2_validate(original_question, original_answer, q_text)
                if not val_res: 
                    print(f"    [Discard] Validation returned None")
                    continue
                
                # Must satisfy both is_clear and is_self_contained
                is_clear = val_res.get('is_clear', False)
                is_self_contained = val_res.get('is_self_contained', False)
                if not is_clear or not is_self_contained:
                    print(f"    [Discard] Validation failed - clear:{is_clear}, self_contained:{is_self_contained}")
                    continue
                
                print(f"    [Validation Passed] clear:{is_clear}, self_contained:{is_self_contained}")
                
                # 3. Solver Verification
                print(f"    [Verifying] Expected: {exp_ans}")
                is_correct = self.step3_verify(q_text, exp_ans)
                if is_correct:
                    cand['validation'] = val_res
                    cand['verified_correct'] = True
                    final_neighbors[cat].append(cand)
                    print(f"    [Keep] Verified question stored under {cat}")
                else:
                    print(f"    [Discard] Solver disagreement for question: {q_text[:60]}...")
        
        # Flatten for output
        flat_list = []
        for cat in self.category_keys:
            flat_list.extend(final_neighbors[cat])
        
        print(f"  [Complete] Generated {len(flat_list)} valid neighbor question(s)")
            
        return {
            "original_question": original_question,
            "original_answer": original_answer,
            "neighbor_questions": flat_list
        }

    # --- Batch Processing ---
    def run(
        self,
        input_file: str,
        output_file: str,
        sample_size: int = None,
        sample_ids: Optional[List[int]] = None,
    ):
        print(f"Loading data: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        items_to_process = []
        for item in data:
            # Support multiple common dataset keys
            q = item.get('question') or item.get('problem') or item.get('original_problem')
            a = item.get('answer') or item.get('original_answer')
            if not q and item.get('metadata', {}).get('question'):
                q = item['metadata']['question']
            if not a and item.get('metadata', {}).get('answer'):
                a = item['metadata']['answer']
            if q and a:
                items_to_process.append({
                    "q": q,
                    "a": a,
                    "meta": item,
                    "id": item.get("id")
                })

        if sample_ids:
            print(f"Filtering by specified IDs: {sample_ids}")
            id_to_item = {item['id']: item for item in items_to_process if item['id'] is not None}
            filtered = []
            for sid in sample_ids:
                match = id_to_item.get(sid)
                if match:
                    filtered.append(match)
                else:
                    print(f"[Warn] ID {sid} not found in dataset or missing 'id' field.")
            items_to_process = filtered
        
        if sample_size:
            items_to_process = items_to_process[:sample_size]
            
        print(f"Processing {len(items_to_process)} items with {self.max_workers} workers...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_item, i['q'], i['a']): i['meta']
                for i in items_to_process
            }
            
            with tqdm(total=len(items_to_process)) as pbar:
                for future in as_completed(futures):
                    meta = futures[future]
                    try:
                        res = future.result()
                        res['metadata'] = meta
                        results.append(res)
                    except Exception as e:
                        print(f"Error: {e}")
                    pbar.update(1)
                    
                    if len(results) % 10 == 0:
                        self._save_json(results, output_file + ".tmp")

        self._save_json(results, output_file)
        if os.path.exists(output_file + ".tmp"):
            os.remove(output_file + ".tmp")
            
        total_nq = sum(len(r['neighbor_questions']) for r in results)
        print(f"Done. Generated {total_nq} valid neighbor questions.")

    def _save_json(self, data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Self-Contained Truth-Anchored Neighbor Questions")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model_name", type=str, default="glm-4-air")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--generation_rounds", type=int, default=1)
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma separated list of entry IDs to process")
    parser.add_argument("--thinking_mode", type=str, default="disabled", help="Zhipu deep-thinking mode (disabled, enabled, ...)")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("Please provide API Key via --api_key or env var ZHIPU_API_KEY")

    generator = NeighborQuestionGenerator(
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_workers=args.max_workers,
        generation_rounds=args.generation_rounds,
        thinking_mode=args.thinking_mode
    )
    
    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(x.strip()) for x in args.sample_ids.split(",") if x.strip().isdigit()]

    generator.run(
        input_file=args.input_file,
        output_file=args.output_file,
        sample_size=args.sample_size,
        sample_ids=sample_ids
    )