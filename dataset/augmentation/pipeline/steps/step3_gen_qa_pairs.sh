#!/bin/bash
set -e

PROVIDER="deepseek"
INPUT_FILE="/disk0/xuhaoming/confidence/dataset/augmentation/result/origin_with_docs.json"
OUTPUT_FILE="/disk0/xuhaoming/confidence/dataset/augmentation/result/origin_with_qa_pairs.json"
MODEL_NAME="DeepSeek-V3.2"
BASE_URL="https://www.dmxapi.cn/v1"
MAX_WORKERS=64
API_CONCURRENCY=64

# Quota parameters
QA_PAIRS_PER_DOC=5
OQ_LEARNING_QA_PAIRS=10
NQ_LEARNING_QA_PAIRS=5
OQ_NQ_COMBINED_QA_PAIRS=4
MAX_NQS_FOR_LEARNING=5

python3 -m pipeline.steps.step3_gen_qa_pairs \
  --provider $PROVIDER \
  --input_file $INPUT_FILE \
  --output_file $OUTPUT_FILE \
  --model_name $MODEL_NAME \
  --base_url $BASE_URL \
  --max_workers $MAX_WORKERS \
  --api_concurrency $API_CONCURRENCY \
  --qa_pairs_per_doc $QA_PAIRS_PER_DOC \
  --oq_learning_qa_pairs $OQ_LEARNING_QA_PAIRS \
  --nq_learning_qa_pairs $NQ_LEARNING_QA_PAIRS \
  --max_nqs_for_learning $MAX_NQS_FOR_LEARNING \
  --oq_nq_combined_qa_pairs $OQ_NQ_COMBINED_QA_PAIRS