import json

SAMPLE_100_PATH = "/disk0/xuhaoming/confidence/dataset/selected_100_samples.json"
OQOA_PATH = "/disk0/xuhaoming/confidence/dataset/augmentation/result/selected_100_samples_OQOA.json"

with open(SAMPLE_100_PATH, "r") as f:
    data = json.load(f)

new_data = [] 
for k, v in data.items():
    for item in v:
        new_data.append({"question": item["original_question"], "answer": item["original_answer"]})

with open(OQOA_PATH, "w") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)
print(f"Extracted {len(new_data)} samples to {OQOA_PATH}")