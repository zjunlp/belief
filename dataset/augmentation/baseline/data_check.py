import json

# Hardcoded output file path
output_file = "/path/to/dataset/selected_100_samples_verified_training_baseline_qa.json"

with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)

from collections import defaultdict

qa_count = defaultdict(int)
for item in data:
    source_id = item.get("source_id")
    if source_id is not None:
        qa_count[source_id] += 1
print(f"Total {len(qa_count)} source_id")
print("source_id does not reach 100 QA pairs:")
for sid, count in qa_count.items():
    if count < 100:
        print(f"source_id: {sid}, QA pairs: {count}")