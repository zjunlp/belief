#!/bin/bash
# =============================================================================
# Level 2 Belief Intervention Pipeline
# =============================================================================
# This script runs the belief intervention evaluation pipeline:
#   1. Retrieve misleading questions
#   2. Convert questions to statements  
#   3. Compute semantic overlap
#   4. Run Asch conformity & source credibility experiments
#   5. Extract entities and compute accuracy
#   6. Generate plots
# =============================================================================

set -e

# =============================================================================
# Configuration - MODIFY THESE PATHS
# =============================================================================

# Input/Output paths
TAG="${TAG:-experiment}"
ORIGIN_DATA_DIR="${ORIGIN_DATA_DIR:-/path/to/level1_output}"   # Level1 output directory
WORK_DATA_DIR="${WORK_DATA_DIR:-./output}"                     # Output directory
HALLUCINATION_FILE="${HALLUCINATION_FILE:-/path/to/misleading_nq.json}"

# Model paths  
TEST_MODEL_PATH="${TEST_MODEL_PATH:-/path/to/your/model}"      # Model to evaluate
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/path/to/judge/model}"   # Judge model (e.g., Qwen2.5-32B-Instruct)

# GPU settings
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# Experiment settings
TEMPERATURE="${TEMPERATURE:-0.7}"
SAMPLE_N="${SAMPLE_N:-10}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

# =============================================================================
# Setup
# =============================================================================

RESULTS_DIR="${WORK_DATA_DIR}/results"
OUTPUT_PLOT_DIR="${RESULTS_DIR}/plot"
mkdir -p "$WORK_DATA_DIR" "$RESULTS_DIR" "$OUTPUT_PLOT_DIR"

# File paths
FILE_ORIGIN="${ORIGIN_DATA_DIR}/${TAG}_responded_score.json"
FILE_HALLU="${WORK_DATA_DIR}/${TAG}_hallu_nq.json"
FILE_CONVERTED="${WORK_DATA_DIR}/${TAG}_hallu_nq_converted.json"
FILE_OVERLAP="${WORK_DATA_DIR}/${TAG}_hallu_nq_semantic_overlap.json"
RESULT_FEWSHOT="${RESULTS_DIR}/test_asch_source.json"
RESULT_FEWSHOT_EXTRACTED="${RESULTS_DIR}/test_asch_source_extracted.json"

echo "=============================================="
echo "Level 2 Belief Intervention Pipeline"
echo "=============================================="
echo "Input:       $FILE_ORIGIN"
echo "Output dir:  $WORK_DATA_DIR"
echo "Model:       $TEST_MODEL_PATH"
echo "Judge model: $JUDGE_MODEL_PATH"
echo "=============================================="

# =============================================================================
# Step 1: Retrieve hallucinations
# =============================================================================
if [ -f "$FILE_HALLU" ]; then
  echo "[Skip] Hallu exists: $FILE_HALLU"
else
  python retrieve_misleading_nq.py \
    --input_file "$FILE_ORIGIN" \
    --output_file "$FILE_HALLU" \
    --hallucination_file "$HALLUCINATION_FILE" \
    --preserve_existing
fi

# =============================================================================
# Step 2: Convert questions to statements
# =============================================================================
if [ -f "$FILE_CONVERTED" ]; then
  echo "[Skip] Converted exists: $FILE_CONVERTED"
else
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python preprocess/cli.py convert-statements \
    --input_file "$FILE_HALLU" \
    --output_file "$FILE_CONVERTED" \
    --model_name "$JUDGE_MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
fi

# =============================================================================
# Step 3: Semantic overlap (entity replacement)
# =============================================================================
if [ -f "$FILE_OVERLAP" ]; then
  echo "[Skip] Overlap exists: $FILE_OVERLAP"
else
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python preprocess/cli.py semantic-overlap \
    --input_file "$FILE_CONVERTED" \
    --output_file "$FILE_OVERLAP" \
    --model_path "$JUDGE_MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
fi

# =============================================================================
# Step 4: Fewshot experiments (Asch + source credibility)
# =============================================================================
if [ -f "$RESULT_FEWSHOT" ]; then
  echo "[Skip] Fewshot exists: $RESULT_FEWSHOT"
else
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python misleading_steering.py \
    --input_file "$FILE_OVERLAP" \
    --output_file "$RESULT_FEWSHOT" \
    --model_path "$TEST_MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --mode all \
    --asch-configs "6,0;5,1;4,2;3,3;2,4;1,5;0,6" \
    --asch-agents 6 \
    --source-levels low medium high \
    --temperature "$TEMPERATURE" \
    --sample-n "$SAMPLE_N" \
    --max-tokens "$MAX_TOKENS"
fi

# =============================================================================
# Step 5: Entity extraction
# =============================================================================
if [ -f "$RESULT_FEWSHOT_EXTRACTED" ]; then
  echo "[Skip] Extracted exists: $RESULT_FEWSHOT_EXTRACTED"
else
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python extract_entities.py \
    --input_file "$RESULT_FEWSHOT" \
    --output_file "$RESULT_FEWSHOT_EXTRACTED" \
    --judge_model_path "$JUDGE_MODEL_PATH" \
    --num_gpu "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
fi

# =============================================================================
# Step 6: Plot statistics
# =============================================================================
if [ -f "$RESULT_FEWSHOT_EXTRACTED" ]; then
  python plot.py \
    --input_file "$RESULT_FEWSHOT_EXTRACTED" \
    --output_dir "$OUTPUT_PLOT_DIR"
else
  echo "[Warn] Plot input missing: $RESULT_FEWSHOT_EXTRACTED"
fi

# =============================================================================
# Done
# =============================================================================
echo "[Completed] Pipeline finished. Outputs:"
echo "  Hallu:        $FILE_HALLU"
echo "  Converted:    $FILE_CONVERTED"
echo "  Overlap:      $FILE_OVERLAP"
echo "  Fewshot:      $RESULT_FEWSHOT"
echo "  Extracted:    $RESULT_FEWSHOT_EXTRACTED"
echo "  Plots:        $OUTPUT_PLOT_DIR"
