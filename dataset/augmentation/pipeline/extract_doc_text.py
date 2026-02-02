import json
import os

MISLEAD_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/selected_100_samples_to_refer_with_docs.json"
OUTPUT_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/text_oqdoc.json"

FACT_TYPE = "origin" # origin or nq

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    if not os.path.exists(MISLEAD_PATH):
        print(f"Error: File not found: {MISLEAD_PATH}")
        return

    mislead_docs = load_json(MISLEAD_PATH)
    
    out = []
    
    for d in mislead_docs:
        # Replicating the logic from merge_qa_doc.py to select valid entries
        k = d.get("id")
        if isinstance(k, int):
            metadata = d.get("metadata", {})
            docs = metadata.get("docs", [])
            for doc in docs:
                if doc.get("fact_type") == FACT_TYPE:
                    content = doc.get("content")
                    if content:
                        out.append({"text": content})

    save_json(OUTPUT_PATH, out)
    print(f"Extracted {len(out)} documents.")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
