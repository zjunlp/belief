#!/bin/bash
set -e

# Configuration
# ==============================================================================
# Base directories
BASE_DIR=$(cd "$(dirname "$0")"; pwd)
SCRIPTS_DIR="$BASE_DIR/scripts"
DATA_DIR="$BASE_DIR/../dataset"

# Input Data Paths (External)
SEED_REFER_FILE="$DATA_DIR/fact_belief_2000_annotated_nq_refined_verified.json"
QA_BASELINE_DATA="$DATA_DIR/selected_100_samples_verified_training_baseline_qa_new.json"
C4_DATA="$DATA_DIR/c4_en_500.json"

# API Configuration (Defaults)
PROVIDER="${PROVIDER:-deepseek}"
API_KEY="${API_KEY:-your_api_key_here}"
BASE_URL="${BASE_URL:-https://www.dmxapi.cn/v1}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V3.2}"

# Intermediate and Output Files
DOC_TYPES_FILE="$DATA_DIR/origin_with_doc_types.json"
DOCS_FILE="$DATA_DIR/selected_100_samples_to_refer_with_docs.json"
STAGE1_FILE="$DATA_DIR/train_dataset_stage1.json"
STAGE2_FILE="$DATA_DIR/train_dataset_stage2.json"

# Execution Settings
MAX_WORKERS=64
API_CONCURRENCY=64

# ==============================================================================

# Ensure directories exist
mkdir -p "$DATA_DIR" 

echo "Starting Reproducible Pipeline..."
echo "Base Directory: $BASE_DIR"

# Step 1: Preprocess Seed Data
echo "[Step 1] download c4 data..."
python3 "$SCRIPTS_DIR/dld_c4.py"

# Step 2: Generate Document Types
echo "[Step 2] Generating document types..."
python3 "$SCRIPTS_DIR/step1_gen_doc_types.py" \
    --provider "$PROVIDER" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --input_file "$SEED_REFER_FILE" \
    --output_file "$DOC_TYPES_FILE" \
    --max_workers "$MAX_WORKERS" \
    --api_concurrency "$API_CONCURRENCY"

# Step 3: Generate Documents
echo "[Step 3] Generating distractor documents..."
python3 "$SCRIPTS_DIR/step2_gen_docs.py" \
    --provider "$PROVIDER" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --input_file "$DOC_TYPES_FILE" \
    --output_file "$DOCS_FILE" \
    --max_workers "$MAX_WORKERS" \
    --api_concurrency "$API_CONCURRENCY"

# Step 4: Stage 1 Merge (Docs + QA)
echo "[Step 4] Merging Stage 1 (Distractor Docs + Baseline QA)..."
python3 "$SCRIPTS_DIR/merge_qa_doc_stage1.py" \
    --mislead_path "$DOCS_FILE" \
    --qa_path "$QA_BASELINE_DATA" \
    --output_path "$STAGE1_FILE"

# Step 5: Stage 2 Merge (Stage 1 + C4)
echo "[Step 5] Merging Stage 2 (Stage 1 + C4 Data)..."
python3 "$SCRIPTS_DIR/merge_qa_doc_stage2.py" \
    --stage1_path "$STAGE1_FILE" \
    --c4_path "$C4_DATA" \
    --output_path "$STAGE2_FILE"

echo "Pipeline completed successfully!"
echo "Final output: $STAGE2_FILE"
