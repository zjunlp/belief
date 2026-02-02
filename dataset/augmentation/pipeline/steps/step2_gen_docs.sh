#!/bin/bash
set -e

PROVIDER="deepseek"
INPUT_FILE="/disk0/xuhaoming/confidence/dataset/augmentation/result/origin_with_doc_types.json"
OUTPUT_FILE="/disk0/xuhaoming/confidence/dataset/augmentation/result/origin_with_docs.json"
MODEL_NAME="DeepSeek-V3.2"
BASE_URL="https://www.dmxapi.cn/v1"
MAX_WORKERS=64
API_CONCURRENCY=64

python3 -m pipeline.steps.step2_gen_docs \
  --provider $PROVIDER \
  --input_file $INPUT_FILE \
  --output_file $OUTPUT_FILE \
  --model_name $MODEL_NAME \
  --base_url $BASE_URL \
  --max_workers $MAX_WORKERS \
  --api_concurrency $API_CONCURRENCY