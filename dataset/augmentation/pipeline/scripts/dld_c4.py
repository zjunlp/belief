from datasets import load_dataset

import os
# 替换为你的实际代理地址（如 http://127.0.0.1:7890）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
# 使用流式加载只获取前500条
dataset = load_dataset(
    "allenai/c4", 
    "en", 
    split="train",
    streaming=True  # 启用流式模式
)
# dataset=load_dataset("microsoft/wiki_qa", split="train")


# 只取前500条
dataset = dataset.take(500)

# 转换为可迭代列表
dataset_list = list(dataset)

# for wikiqa
# dataset_list = [item for item in dataset_list if item["label"]==1]

# 保存到文件
import json
with open("../../dataset/c4_en_500.json", "w", encoding="utf-8") as f:
    json.dump(dataset_list, f, ensure_ascii=False, indent=2)
