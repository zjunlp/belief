#!/bin/bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEVICES}}

# ===== User-configurable knobs =====
MODEL_PATH=${MODEL_PATH:-/disk0/share/models/Qwen2.5-32B-Instruct}
TP_SIZE=${TP_SIZE:-8}
GPU_MEM=${GPU_MEM:-0.9}
BATCH_SIZE=${BATCH_SIZE:-16}
FEWSHOT=${FEWSHOT:-3}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-1024}
OUT_DIR=${OUT_DIR:-/disk0/xuhaoming/confidence/analysis/eval_runs}
TASKS=${TASKS:-bbh}                                   # space-separated list
LORA_ADAPTERS=${LORA_ADAPTERS:-""}                    # space-separated list; empty entry means base model
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-}
GEN_KWARGS=${GEN_KWARGS:-max_gen_toks=${MAX_GEN_TOKS}}
LOG_SAMPLES=${LOG_SAMPLES:-true}

mkdir -p "${OUT_DIR}"
base_model_name=$(basename "${MODEL_PATH}")

# parse tasks and loras
read -r -a TASK_LIST <<< "${TASKS}"
if [[ ${#TASK_LIST[@]} -eq 0 ]]; then
  echo "No tasks specified via TASKS env."
  exit 1
fi

if [[ -z "${LORA_ADAPTERS// }" ]]; then
  LORA_ADAPTERS=""
fi
read -r -a LORA_LIST <<< "${LORA_ADAPTERS}"
if [[ ${#LORA_LIST[@]} -eq 0 ]]; then
  LORA_LIST=("")
fi

echo "Starting lm-eval runs..."
echo "Model path: ${MODEL_PATH}"
echo "Tasks: ${TASKS}"
echo "Lora adapters: ${LORA_ADAPTERS:-<none>}"
echo "Output dir: ${OUT_DIR}"

for lora_adapter_dir in "${LORA_LIST[@]}"; do
  if [[ -n "${lora_adapter_dir}" ]]; then
    lora_adapter_name=$(basename "${lora_adapter_dir}")
    model_args="pretrained=${MODEL_PATH},enable_lora=True,lora_local_path=${lora_adapter_dir},tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=${GPU_MEM},trust_remote_code=True,dtype=bfloat16,max_lora_rank=256"
  else
    lora_adapter_name="nolora"
    model_args="pretrained=${MODEL_PATH},enable_lora=False,tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=${GPU_MEM},trust_remote_code=True,dtype=bfloat16"
  fi

  for task in "${TASK_LIST[@]}"; do
    output_path="${OUT_DIR}/${task}_${base_model_name}_${lora_adapter_name}_vllm.json"
    cmd=(lm_eval
      --model vllm
      --model_args "${model_args}"
      --tasks "${task}"
      --num_fewshot "${FEWSHOT}"
      --batch_size "${BATCH_SIZE}"
      --gen_kwargs "${GEN_KWARGS}"
      --output_path "${output_path}"
    )

    if [[ "${APPLY_CHAT_TEMPLATE}" == "true" ]]; then
      cmd+=(--apply_chat_template)
    fi
    if [[ -n "${SYSTEM_PROMPT}" ]]; then
      cmd+=(--system_instruction "${SYSTEM_PROMPT}")
    fi
    if [[ "${LOG_SAMPLES}" == "true" ]]; then
      cmd+=(--log_samples)
    fi

    echo "Running task=${task} lora=${lora_adapter_name}"
    "${cmd[@]}"
  done
done

echo "Evaluation completed, results saved to: ${OUT_DIR}"
