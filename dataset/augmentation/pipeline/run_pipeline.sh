#!/bin/bash
set -euo pipefail

# Pipeline run script (using new modular structure)

# Configuration
DIR="/disk0/xuhaoming/confidence/dataset/augmentation/result"
INPUT_FILE="${DIR}/selected_100_samples_to_refer.json"

# General Configuration
PROVIDER="deepseek"
BASE_URL="https://www.dmxapi.cn/v1"
MODEL_NAME="DeepSeek-V3.2"
MAX_WORKERS=64
API_CONCURRENCY=64

# Step Configuration
MAX_TYPES=6
QA_PAIRS_PER_DOC=5
OQ_LEARNING_QA_PAIRS=10
NQ_LEARNING_QA_PAIRS=5
OQ_NQ_COMBINED_QA_PAIRS=4
MAX_NQS_FOR_LEARNING=5

# Ensure directory exists
mkdir -p "${DIR}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Pipeline Directory: ${PIPELINE_DIR}"
echo "Provider: ${PROVIDER} | Base URL: ${BASE_URL} | Model: ${MODEL_NAME}"
echo "Workers: ${MAX_WORKERS} | API Concurrency: ${API_CONCURRENCY}"
echo "Note: Using API key from environment (DEEPSEEK_API_KEY/OPENAI_API_KEY or ZHIPU_API_KEY)."
echo "=============================================="

# Run pipeline (Python version)
cd "${PIPELINE_DIR}"
python -m pipeline.run_pipeline \
    --input_file "${INPUT_FILE}" \
    --output_dir "${DIR}" \
    --provider "${PROVIDER}" \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL_NAME}" \
    --max_workers "${MAX_WORKERS}" \
    --api_concurrency "${API_CONCURRENCY}" \
    --max_types "${MAX_TYPES}" \
    --max_nqs_for_learning "${MAX_NQS_FOR_LEARNING}" \
    --qa_pairs_per_doc "${QA_PAIRS_PER_DOC}" \
    --oq_learning_qa_pairs "${OQ_LEARNING_QA_PAIRS}" \
    --nq_learning_qa_pairs "${NQ_LEARNING_QA_PAIRS}" \
    --oq_nq_combined_qa_pairs "${OQ_NQ_COMBINED_QA_PAIRS}"

echo ""
echo "All steps done in ${DIR}."