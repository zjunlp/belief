import json
from collections import defaultdict

MISLEAD_PATH_1 = "/disk0/xuhaoming/confidence/dataset/augmentation/result/train_dataset_doc_mislead_1000.json"
MISLEAD_PATH_2 = "/disk0/xuhaoming/confidence/dataset/augmentation/result/train_dataset_doc_nq_mislead_1000.json"
QA_PATH = "/disk0/xuhaoming/confidence/dataset/selected_100_samples_verified_training_baseline_qa_new.json"
TYPES_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/selected_100_samples_to_refer_with_types.json"
OUTPUT_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/train_dataset_mislead+nq+qa_stage1.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def preview(name, data, n=3):
    print(name, ":")
    for i in range(min(n, len(data))):
        print(data[i])

def group_docs_by_idx(doc_data):
    g = defaultdict(list)
    for d in doc_data:
        k = d.get("idx")
        if isinstance(k, int):
            g[k].append(d)
    return g

def build_src_index_to_id(types_data):
    m = {}
    for i, s in enumerate(types_data):
        v = s.get("id")
        if isinstance(v, int):
            m[i] = v
    return m

def main():
    mislead_docs_1 = load_json(MISLEAD_PATH_1)
    mislead_docs_2 = load_json(MISLEAD_PATH_2)
    qa_items = load_json(QA_PATH)
    types_items = load_json(TYPES_PATH)

    # preview("mislead_1.json", mislead_docs_1, 2)
    # preview("mislead_2.json", mislead_docs_2, 2)
    # preview("qa.json", qa_items, 2)
    # preview("types.json", types_items, 2)

    docs_by_idx_1 = group_docs_by_idx(mislead_docs_1)
    docs_by_idx_2 = group_docs_by_idx(mislead_docs_2)
    src_to_id = build_src_index_to_id(types_items)

    qa_by_id = defaultdict(list)
    bad_qa = 0
    for qa in qa_items:
        si = qa.get("src_index")
        if not isinstance(si, int):
            continue
        idx = src_to_id.get(si)
        if not isinstance(idx, int):
            bad_qa += 1
            continue
        q = qa.get("question")
        a = qa.get("answer")
        if isinstance(q, str) and isinstance(a, str):
            qa_by_id[idx].append({"question": q, "answer": a})

    out = []
    for t in types_items:
        idx = t.get("id")
        if not isinstance(idx, int):
            continue
        oq = t.get("original_question")
        oa = t.get("original_answer")
        if not isinstance(oq, str):
            oq = (t.get("metadata") or {}).get("question")
        if not isinstance(oa, str):
            oa = (t.get("metadata") or {}).get("answer")
        d1 = []
        d2 = []
        for d in docs_by_idx_1.get(idx, []):
            s = d.get("distractor_doc")
            if isinstance(s, str) and s.strip():
                d1.append(s)
        for d in docs_by_idx_2.get(idx, []):
            s = d.get("distractor_doc")
            if isinstance(s, str) and s.strip():
                d2.append(s)
        item = {
            "idx": idx,
            "question": oq,
            "answer": oa,
            "qa_variants": qa_by_id.get(idx, []),
            "distractor_docs_1": d1,
            "distractor_docs_2": d2,
        }
        out.append(item)

    save_json(OUTPUT_PATH, out)
    print("Total samples:", len(out))
    print("QA count that could not be mapped to idx:", bad_qa)
    print("Saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
