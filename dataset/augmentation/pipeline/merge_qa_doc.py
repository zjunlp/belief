import json
import random
from collections import defaultdict

MISLEAD_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/selected_100_samples_to_refer_with_docs.json"
QA_PATH = "/disk0/xuhaoming/confidence/dataset/selected_100_samples_verified_training_baseline_qa_new.json"
# QA_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/selected_100_samples_verified_training_baseline_qa0.json"
TYPES_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/selected_100_samples_to_refer_with_types.json"
OUTPUT_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/merged_qa_nqdoc.json"

FACT_TYPE = "nq" # origin or nq

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def preview(name, data, n=3):
    print(name, "samples:")
    for i in range(min(n, len(data))):
        print(data[i])

def build_id_to_src_index(types_data):
    m = {}
    for i, s in enumerate(types_data):
        m[s.get("id")] = i
    return m

def group_qa_by_src_index(qa_data):
    g = defaultdict(list)
    for x in qa_data:
        si = x.get("source_id")
        if isinstance(si, int):
            g[si].append(x)
    return g

def group_docs_by_idx(doc_data):
    g = defaultdict(list)
    for d in doc_data:
        k = d.get("id")
        if isinstance(k, int):
            # Extract docs from metadata
            metadata = d.get("metadata", {})
            docs = metadata.get("docs", [])
            for doc in docs:
                if doc.get("fact_type") == FACT_TYPE:
                    g[k].append(doc)
    return g

def main():
    mislead_docs = load_json(MISLEAD_PATH)
    qa_items = load_json(QA_PATH)
    types_items = load_json(TYPES_PATH)
    n = 100

    # preview("mislead_1000.json", mislead_docs, 3)
    # preview("qa_new.json", qa_items, 3)

    id_to_src = build_id_to_src_index(types_items)
    qa_by_src = group_qa_by_src_index(qa_items)
    docs_by_idx = group_docs_by_idx(mislead_docs)
    print(f"len(docs_by_idx)={len(docs_by_idx)}")
    print(f"len(qa_by_src)={len(qa_by_src)}")

    out = []
    matched = 0
    for idx, docs in docs_by_idx.items():
        # src = id_to_src.get(idx)
        # breakpoint()
        
        src = idx
        if src is None:
            continue
        qas = qa_by_src.get(src, [])
        if not qas or not docs:
            print(f"Unmatched: idx={idx}, len(qas)={len(qas)}, len(docs)={len(docs)}")
            continue
        for i in range(n):
            doc = random.choice(docs)
            qa = random.choice(qas)
            # Use doc["content"] instead of doc["distractor_doc"]
            q = doc["content"] + "\n" + qa["question"]
            a = qa["answer"]    
            out.append({"question": q, "answer": a})
        matched += 1

    save_json(OUTPUT_PATH, out)
    print("Matched idx count:", matched)
    print("Total generated samples:", len(out))
    print("Saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
