#!/bin/bash
set -euo pipefail

# ===== User-configurable knobs =====
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}                 # GPU ids, comma-separated
MASTER_PORT=${MASTER_PORT:-19123}
MODEL_FAMILY=${MODEL_FAMILY:-Qwen2.5-32B-Instruct}
DATA_PATH=${DATA_PATH:-/disk0/xuhaoming/confidence/dataset/selected_100_samples_training.json}
LR=${LR:-1e-4}
NUM_EPOCHS=${NUM_EPOCHS:-25}
BATCH_SIZE=${BATCH_SIZE:-4}                          # per-device batch size
GRAD_ACCUM=${GRAD_ACCUM:-2}
DS_CONFIG=${DS_CONFIG:-/disk0/xuhaoming/confidence/baselines/finetune/config/ds_z0_config.json}
TRAINING_METHOD=${TRAINING_METHOD:-sft}              # e.g., sft / mixsft
SAVE_NUMS=${SAVE_NUMS:-10}
EXTRA_ARGS=${EXTRA_ARGS:-}                           # e.g., "max_length=1024 doc_data_path=/path"

# ===== Derived paths =====
DATA_NAME=$(basename "${DATA_PATH%.*}")
SAVE_DIR=${SAVE_DIR:-/disk0/xuhaoming/confidence/baselines/finetune/checkpoints/${MODEL_FAMILY}_lora_trl_${TRAINING_METHOD}_lr${LR}_${NUM_EPOCHS}e_${DATA_NAME}}

# ===== Utility: count GPUs =====
IFS=',' read -ra __gpu_list <<< "${DEVICES}"
NPROC_PER_NODE=0
for __gpu in "${__gpu_list[@]}"; do
  __gpu_trimmed=${__gpu// /}
  if [[ -n "${__gpu_trimmed}" ]]; then
    ((NPROC_PER_NODE++))
  fi
done
unset __gpu __gpu_trimmed __gpu_list
if [[ ${NPROC_PER_NODE} -eq 0 ]]; then
  NPROC_PER_NODE=1
fi

echo "Starting TRL fine-tuning..."
echo "Model: ${MODEL_FAMILY}"
echo "Data: ${DATA_PATH}"
echo "Save dir: ${SAVE_DIR}"
echo "Using devices: ${DEVICES} (nproc_per_node=${NPROC_PER_NODE})"

CUDA_VISIBLE_DEVICES=${DEVICES} torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_port=${MASTER_PORT} \
  ../finetune/train.py \
  --config-name=sft_lora \
  batch_size=${BATCH_SIZE} \
  gradient_accumulation_steps=${GRAD_ACCUM} \
  model_family=${MODEL_FAMILY} \
  lr=${LR} \
  num_epochs=${NUM_EPOCHS} \
  data_path=${DATA_PATH} \
  save_dir=${SAVE_DIR} \
  save_nums=${SAVE_NUMS} \
  ds_config=${DS_CONFIG} \
  training_method=${TRAINING_METHOD} \
  ${EXTRA_ARGS}

