#!/usr/bin/env python3
import json
import argparse
import sys
from collections import defaultdict

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Merge: Combine distractor docs with QA pairs.")
    parser.add_argument("--mislead_path", type=str, required=True, help="Path to docs/mislead file (from Step 2)")
    parser.add_argument("--qa_path", type=str, required=True, help="Path to QA baseline file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output Stage 1 JSON file")
    args = parser.parse_args()

    try:
        docs_items = load_json(args.mislead_path)
        qa_items = load_json(args.qa_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # Group QA by source_id
    qa_by_id = defaultdict(list)
    for qa in qa_items:
        # Note: using source_id as it matches the id in the docs file
        si = qa.get("source_id")
        if isinstance(si, int):
            q = qa.get("question")
            a = qa.get("answer")
            if isinstance(q, str) and isinstance(a, str):
                qa_by_id[si].append({"question": q, "answer": a})

    out = []
    
    for item in docs_items:
        idx = item.get("id")
        if not isinstance(idx, int):
            continue

        oq = item.get("original_question")
        oa = item.get("original_answer")
        
        # Fallback if not at top level
        if not isinstance(oq, str):
            oq = (item.get("metadata") or {}).get("question")
        if not isinstance(oa, str):
            oa = (item.get("metadata") or {}).get("answer")

        d1 = []
        d2 = []
        
        # Extract docs from metadata
        metadata = item.get("metadata", {})
        docs = metadata.get("docs", [])
        
        for doc in docs:
            content = doc.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
                
            fact_type = doc.get("fact_type")
            if fact_type == "origin":
                d1.append(content)
            elif fact_type == "nq":
                d2.append(content)

        output_item = {
            "idx": idx,
            "question": oq,
            "answer": oa,
            "qa_variants": qa_by_id.get(idx, []),
            "distractor_docs_1": d1,
            "distractor_docs_2": d2,
        }
        out.append(output_item)

    save_json(args.output_path, out)
    print("Total samples:", len(out))
    print("Saved to:", args.output_path)

if __name__ == "__main__":
    main()
