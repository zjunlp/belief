#!/bin/bash

input_file="/path/to/dataset/selected_100_samples_to_refer_verified.json"
output_file="/path/to/dataset/selected_100_samples_verified_training_baseline_qa_new.json"
provider="deepseek"
api_key=""
base_url="https://www.dmxapi.cn/v1"
model_name="DeepSeek-V3.2"
api_concurrency=64
max_workers=64
sample_size=4
echo "Starting baseline QA generation..."
echo "Input: $input_file"
echo "Output: $output_file"
echo "Model: $model_name"

python3 -m dataset.augmentation.baseline.generate_baseline_qa_new \
  --input_file $input_file \
  --output_file $output_file \
  --provider  $provider \
  --model_name $model_name \
  --api_concurrency $api_concurrency \
  --api_key $api_key \
  --base_url $base_url \
  --max_workers $max_workers \
  --dedup \
  --include_meta
  