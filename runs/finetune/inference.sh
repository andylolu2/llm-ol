#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/finetune/v6
step=10000
model=$exp_dir/train/checkpoint-$step

echo "Running inference on $model"

if [ ! -d "$model/merged" ]; then
    python llm_ol/experiments/llm/finetune/export_model.py \
        --checkpoint_dir $model
fi

python llm_ol/experiments/llm/finetune/inference.py \
    --test_dataset out/experiments/llm/v2/eval_dataset.jsonl \
    --model $model/merged \
    --output_dir $exp_dir/$step
