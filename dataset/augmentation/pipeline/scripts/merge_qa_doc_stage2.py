#!/usr/bin/env python3
import json
import random
import argparse
import sys

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_index(stage1_items):
    idx_map = {}
    for it in stage1_items:
        idx = it.get("idx")
        q = it.get("question")
        a = it.get("answer")
        d1 = it.get("distractor_docs_1") or []
        d2 = it.get("distractor_docs_2") or []
        if not isinstance(idx, int):
            continue
        if idx not in idx_map:
            idx_map[idx] = {"qa_variants": [], "docs1": [], "docs2": []}
        if isinstance(q, str) and isinstance(a, str):
            idx_map[idx]["qa_variants"].append({"question": q, "answer": a})
        for s in d1:
            if isinstance(s, str) and s.strip():
                idx_map[idx]["docs1"].append(s)
        for s in d2:
            if isinstance(s, str) and s.strip():
                idx_map[idx]["docs2"].append(s)
    for idx, v in idx_map.items():
        seen = set()
        qas = []
        for qa in v["qa_variants"]:
            k = (qa.get("question"), qa.get("answer"))
            if isinstance(k[0], str) and isinstance(k[1], str) and k not in seen:
                seen.add(k)
                qas.append(qa)
        v["qa_variants"] = qas
        v["docs1"] = list(dict.fromkeys(v["docs1"]))
        v["docs2"] = list(dict.fromkeys(v["docs2"]))
    return idx_map

def sample_pairs(qa_list, texts, n):
    res = []
    if not qa_list or not texts or n <= 0:
        return res
    for _ in range(n):
        qa = random.choice(qa_list)
        t = random.choice(texts)
        res.append({"text": t, "question": qa["question"], "answer": qa["answer"]})
    return res

def sample_c4_pairs(qa_list, c4_texts, n):
    res = []
    if not qa_list or not c4_texts or n <= 0:
        return res
    if len(c4_texts) >= n:
        texts = random.sample(c4_texts, n)
        for t in texts:
            qa = random.choice(qa_list)
            res.append({"text": t, "question": qa["question"], "answer": qa["answer"]})
    else:
        for _ in range(n):
            t = random.choice(c4_texts)
            qa = random.choice(qa_list)
            res.append({"text": t, "question": qa["question"], "answer": qa["answer"]})
    return res

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Merge: Mix Stage 1 data with C4 data.")
    parser.add_argument("--stage1_path", type=str, required=True, help="Path to Stage 1 JSON file")
    parser.add_argument("--c4_path", type=str, required=True, help="Path to C4 data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output Stage 2 JSON file")
    args = parser.parse_args()

    random.seed(42)
    
    try:
        stage1_items = load_json(args.stage1_path)
        c4_items = load_json(args.c4_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    c4_texts = []
    for x in c4_items:
        s = x.get("text")
        if isinstance(s, str) and s.strip():
            c4_texts.append(s)
            
    idx_map = build_index(stage1_items)
    out = []
    for idx, v in idx_map.items():
        qa = v["qa_variants"]
        d1 = v["docs1"]
        d2 = v["docs2"]
        # oq part
        # out.extend(sample_pairs(qa, d1, 100))
        # nq part
        out.extend(sample_pairs(qa, d2, 100))
        # c4 part
        out.extend(sample_c4_pairs(qa, c4_texts, 200))
        
    random.shuffle(out)
    save_json(args.output_path, out)
    print("Total samples:", len(out))
    print("Saved to:", args.output_path)

if __name__ == "__main__":
    main()
