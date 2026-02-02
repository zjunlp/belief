#!/bin/bash
set -e

# Parameter configuration
PROVIDER="deepseek"
API_KEY="YOUR_API_KEY"
INPUT_FILE="/disk0/xuhaoming/confidence/dataset/augmentation/result/origin.json"
OUTPUT_FILE="/disk0/xuhaoming/confidence/dataset/augmentation/result/origin_with_doc_types.json"
MODEL_NAME="DeepSeek-V3.2"
BASE_URL="https://www.dmxapi.cn/v1"
MAX_WORKERS=64
API_CONCURRENCY=64
MAX_TYPES=6

python3 -m pipeline.steps.step1_gen_doc_types \
  --provider $PROVIDER \
  --input_file $INPUT_FILE \
  --output_file $OUTPUT_FILE \
  --model_name $MODEL_NAME \
  --base_url $BASE_URL \
  --max_workers $MAX_WORKERS \
  --api_concurrency $API_CONCURRENCY \
  --max_types $MAX_TYPES