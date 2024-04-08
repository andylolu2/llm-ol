#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

model=out/experiments/finetune/v3/out/checkpoint-1000

if [ ! -d "$model/merged" ]; then
    echo "Exporting model to $model/merged"
    python llm_ol/experiments/llm/finetune/export_model.py \
        --checkpoint_dir $model
fi

python llm_ol/experiments/llm/finetune/inference.py \
    --test_dataset out/experiments/llm/v1/test_dataset.jsonl \
    --model $model/merged \
    --output_dir out/experiments/finetune/v3
