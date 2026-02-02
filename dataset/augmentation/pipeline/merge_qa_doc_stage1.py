import json
from collections import defaultdict

MISLEAD_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/selected_100_samples_to_refer_with_docs.json"
QA_PATH = "/disk0/xuhaoming/confidence/dataset/selected_100_samples_verified_training_baseline_qa_new.json"
OUTPUT_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/train_dataset_distill+oqnqlongqa_stage1.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    docs_items = load_json(MISLEAD_PATH)
    qa_items = load_json(QA_PATH)

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
    bad_qa = 0 # Count items where we might miss QAs if we were strictly mapping, but here we just check if we have QAs
    
    for item in docs_items:
        idx = item.get("id")
        if not isinstance(idx, int):
            continue

        oq = item.get("original_question")
        oa = item.get("original_answer")
        
        # Fallback if not at top level (though they seem to be there)
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

    save_json(OUTPUT_PATH, out)
    print("Total samples:", len(out))
    print("Saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
