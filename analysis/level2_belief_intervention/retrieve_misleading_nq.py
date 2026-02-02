#!/usr/bin/env python3

import json
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Ensure project root on path for `utils` imports (project root: confidence/analysis)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import load_json, save_json

class HallucinationRetriever:
    def __init__(self, hallucination_file: str):
        self.hallucination_data = self._load_hallucination_data(hallucination_file)
    
    def _load_hallucination_data(self, path: str) -> Dict[tuple, Dict]:
        print(f"[INFO] Loading hallucination data from {path}...")
        data = load_json(path)
        
        # Build lookup index
        # Key: (original_question, original_answer) -> item
        # Normalize keys by stripping whitespace
        lookup = {}
        for item in data:
            q = item.get('original_question', '').strip()
            a = item.get('original_answer', '').strip()
            if q and a:
                lookup[(q, a)] = item
        
        print(f"[INFO] Indexed {len(lookup)} hallucination entries.")
        return lookup

    def process_item(self, original_question: str, original_answer: str) -> Optional[Dict[str, Any]]:
        key = (original_question.strip(), original_answer.strip())
        if key not in self.hallucination_data:
            return None

        entry = self.hallucination_data[key]

        return {
            "original_question": original_question,
            "original_answer": original_answer,
            "misleading_entity": entry.get('target_hallucination', ""),
            "misleading_neighbor_questions": entry.get('neighbor_questions', []),
        }

    def run(self, input_file: str, output_file: str, sample_size: int = None, sample_ids: Optional[List[int]] = None, preserve_existing: bool = True):
        print(f"[INFO] Loading input file {input_file}...")
        data = load_json(input_file)

        items = []
        for item in data:
            orig_neighbors = item.get('neighbor_questions', [])
            q = item.get('question') or item.get('problem') or item.get('original_problem') or item.get('original_question')
            a = item.get('answer') or item.get('original_answer')
            if q and a:
                meta = item.pop('metadata', {})
                items.append({
                    "q": q,
                    "a": a,
                    "meta": meta,
                    "id": item.get("id"),
                    "orig_neighbors": orig_neighbors,
                    "belief_result": item.get('belief_result', {}),
                    "oq_responses": item["confidence"]["consistency_confidence"]["all_answers"],
                    "oq_entities": item["confidence"]["consistency_confidence"]["all_entities"],
                })

        if sample_ids:
            items = [x for x in items if x['id'] in sample_ids]
        if sample_size:
            items = items[:sample_size]
        
        print(f"[INFO] Processing {len(items)} items...")
        results = []
        
        found_count = 0
        missing_count = 0
        
        with tqdm(total=len(items)) as pbar:
            for item in items:
                res = self.process_item(item['q'], item['a'])
                res['belief_result'] = item['belief_result']
                res['oq_responses'] = item['oq_responses']
                res['oq_entities'] = item['oq_entities']
                meta = item['meta']
                orig_neighbors = item.get("orig_neighbors", [])
                
                if res:
                    found_count += 1
                    # Merge Logic
                    # Always preserve original neighbors if present (input side)
                    if orig_neighbors:
                        res['original_neighbor_questions'] = orig_neighbors
                    elif 'neighbor_questions' in meta:
                        res['original_neighbor_questions'] = meta.get('neighbor_questions', [])
                    if preserve_existing:
                        for key in ['id', 'metadata']:
                            if key in meta and key not in res:
                                res[key] = meta[key]
                    else:
                        # still carry metadata for downstream consistency
                        if 'metadata' in meta:
                            res['metadata'] = meta['metadata']
                    
                    res['metadata'] = meta
                    results.append(res)
                else:
                    missing_count += 1
                    # Handle missing items by recording an error, consistent with original script failure mode
                    error_res = {
                        "original_question": item['q'],
                        "original_answer": item['a'],
                        "error": "not_found_in_hallucination_file",
                        "metadata": meta
                    }
                    if orig_neighbors:
                        error_res["original_neighbor_questions"] = orig_neighbors
                    if 'id' in meta:
                        error_res['id'] = meta['id']
                    results.append(error_res)
                
                pbar.update(1)
                if len(results) % 100 == 0:
                    save_json(results, output_file + ".tmp")

        save_json(results, output_file)
        if os.path.exists(output_file + ".tmp"):
            os.remove(output_file + ".tmp")
        
        print(f"[Done] Saved {len(results)} items to {output_file}")
        print(f"Found: {found_count}, Missing: {missing_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve pre-generated hallucination questions based on original question/answer pairs.")
    parser.add_argument("--input_file", required=True, help="The original dataset file to process")
    parser.add_argument("--output_file", required=True, help="Where to save the merged results")
    parser.add_argument("--hallucination_file", required=True, help="Path to the pre-generated hallucination data")
    
    parser.add_argument("--sample_size", type=int, default=None, help="Limit number of items to process")
    parser.add_argument("--sample_ids", default=None, help="Comma-separated list of IDs to process")
    parser.add_argument("--preserve_existing", action="store_true", help="Preserve existing fields from input metadata")
    
    args = parser.parse_args()
    
    retriever = HallucinationRetriever(args.hallucination_file)
    
    sample_ids = [int(x) for x in args.sample_ids.split(",")] if args.sample_ids else None
    retriever.run(args.input_file, args.output_file, args.sample_size, sample_ids, args.preserve_existing)
