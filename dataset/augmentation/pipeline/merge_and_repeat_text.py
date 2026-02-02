import json
import random
import os

PATH_OQ = "/disk0/xuhaoming/confidence/dataset/augmentation/result/text_oqdoc.json"
PATH_NQ = "/disk0/xuhaoming/confidence/dataset/augmentation/result/text_nqdoc.json"
OUTPUT_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/merged_repeated_text.json"

REPEAT_TIMES = 10

def load_json(path):
    print(f"Loading {path}...")
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist. Returning empty list.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    print(f"Saving to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    data_oq = load_json(PATH_OQ)
    data_nq = load_json(PATH_NQ)
    
    print(f"Loaded {len(data_oq)} items from OQ.")
    print(f"Loaded {len(data_nq)} items from NQ.")
    
    merged = data_oq + data_nq
    print(f"Merged count: {len(merged)}")
    
    final_data = merged * REPEAT_TIMES
    print(f"After repeating {REPEAT_TIMES} times: {len(final_data)}")
    
    random.shuffle(final_data)
    print("Shuffled data.")
    
    save_json(OUTPUT_PATH, final_data)
    print("Done.")

if __name__ == "__main__":
    main()
