# load c4 qa , doc qa and shuffle them

import json
with open("/disk0/xuhaoming/confidence/dataset/augmentation/result/merged_qa_oqdoc.json", "r", encoding="utf-8") as f:
    oqdoc = json.load(f)
with open("/disk0/xuhaoming/confidence/dataset/augmentation/result/merged_qa_nqdoc.json", "r", encoding="utf-8") as f:
    nqdoc = json.load(f)

with open("/disk0/xuhaoming/confidence/dataset/augmentation/result/merged_longqa_c4.json", "r", encoding="utf-8") as f:
    c4doc = json.load(f)

results = []
# results.extend(oqdoc)
results.extend(nqdoc)
results.extend(c4doc)

import random
random.shuffle(results)

with open("/disk0/xuhaoming/confidence/dataset/augmentation/result/train_dataset_nqqa+c4qa.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
