#!/bin/bash

INPUT_FILE="/path/to/dataset/fact_belief_2000_annotated_nq_refined_verified.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
CUSTOM_SUFFIX=${1:-"qwen3instbase_fulldata"}
if [ -n "$CUSTOM_SUFFIX" ]; then
  OUTPUT_DIR="/path/to/analysis/data${TIMESTAMP}_${CUSTOM_SUFFIX}"
else
  OUTPUT_DIR="/path/to/analysis/data${TIMESTAMP}"
fi
BASE_MODEL="/path/to/model/Qwen3-30B-A3B-Instruct-2507"
JUDGE_MODEL="/path/to/model/Qwen2.5-32B-Instruct/"
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.90
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OQ_NUM_SAMPLES=30
OQ_TEMPERATURE=0.7
NQ_NUM_SAMPLES=30
NQ_TEMPERATURE=0.7
mkdir -p "$OUTPUT_DIR"

base_name=$(basename "$BASE_MODEL")
echo "Testing base model: $base_name"

output_file="$OUTPUT_DIR/${base_name}_fact_belief_2000_nq_fulldata.json"
if [ ! -f "$output_file" ]; then
    python gen_oq_dual_model.py \
        --inference_model_name $BASE_MODEL \
        --entity_model_name $JUDGE_MODEL \
        --inference_tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --entity_tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --inference_gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
        --entity_gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
        --input_file $INPUT_FILE \
        --output_file $output_file \
        --num_samples $OQ_NUM_SAMPLES
else
    echo "Result file already exists: $output_file"
fi


output_file_responded="$OUTPUT_DIR/${base_name}_fact_belief_2000_nq_responded_fulldata.json"
if [ ! -f "$output_file_responded" ]; then
    python gen_nq.py \
      --model_name $BASE_MODEL \
      --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
      --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
      --input_file $output_file \
      --output_file $output_file_responded \
      --num_samples $NQ_NUM_SAMPLES \
      --temperature $NQ_TEMPERATURE
else
    echo "Result file already exists: $output_file_responded"
fi


output_file_responded_score="$OUTPUT_DIR/${base_name}_fact_belief_2000_nq_responded_score_fulldata.json"
output_file_responded_score_valid="$OUTPUT_DIR/${base_name}_fact_belief_2000_nq_responded_valid_fulldata.json"
output_file_responded_score_invalid="$OUTPUT_DIR/${base_name}_fact_belief_2000_nq_responded_invalid_fulldata.json"
python calc_belief_score.py \
  --input_file $output_file_responded \
  --output_file $output_file_responded_score \
  --output_valid $output_file_responded_score_valid \
  --output_invalid $output_file_responded_score_invalid \
  --analysis_output_dir $OUTPUT_DIR \
  --match_type loose \
  --quantile 0.25 \
  --balance_by_py \
  --num_bins 20