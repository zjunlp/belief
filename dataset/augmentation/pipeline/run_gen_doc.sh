#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Script to run the Strict Document Generation Step
# -----------------------------------------------------------------------------

# Environment & Interpreter
PYTHON_BIN="/home/xuhaoming/miniforge3/envs/confidence/bin/python"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_SCRIPT="${SCRIPT_DIR}/steps/mislead_doc_nq_generation.py"

# Default Data Configuration
# You can override these variables or pass them as arguments if you modify the script
INPUT_FILE="/disk0/xuhaoming/confidence/dataset/augmentation/result/selected_100_samples_to_refer_with_types.json"
OUTPUT_FILE="/disk0/xuhaoming/confidence/dataset/augmentation/result/train_dataset_doc_nq_mislead_1000.json"


# Model Configuration
PROVIDER="deepseek"
BASE_URL="https://www.dmxapi.cn/v1"
MODEL_NAME="DeepSeek-V3.2"
MAX_WORKERS=64
API_CONCURRENCY=64

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

if [ ! -f "$STEP_SCRIPT" ]; then
    echo "Error: Python script not found at $STEP_SCRIPT"
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python interpreter not executable at $PYTHON_BIN"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

echo "=============================================="
echo "Starting Strict Document Generation"
echo "----------------------------------------------"
echo "Date: $(date)"
echo "Interpreter: ${PYTHON_BIN}"
echo "Script:      ${STEP_SCRIPT}"
echo "Input File:  ${INPUT_FILE}"
echo "Output File: ${OUTPUT_FILE}"
echo "Model:       ${MODEL_NAME} (${PROVIDER})"
echo "Workers:     ${MAX_WORKERS}"
echo "=============================================="

# Check if input file exists before running
if [ ! -f "$INPUT_FILE" ]; then
    echo "Warning: Input file does not exist: $INPUT_FILE"
    echo "Please ensure Step 1 (gen_types) has been run successfully."
    # We don't exit here to allow the python script to handle it or show help if args are wrong,
    # but practically the python script will fail. 
    # Let's exit to be safe unless user wants to see help.
    echo "Exiting."
    exit 1
fi

"$PYTHON_BIN" "$STEP_SCRIPT" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================="
    echo "Success! Output generated at:"
    echo "$OUTPUT_FILE"
    echo "=============================================="
else
    echo "=============================================="
    echo "Failed with exit code $EXIT_CODE"
    echo "=============================================="
fi

exit $EXIT_CODE
