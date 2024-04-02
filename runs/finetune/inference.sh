#!/bin/bash
set -e

export HF_HUB_CACHE=out/models

model=out/experiments/finetune/v3/out/final

if [ ! -d "$model/merged" ]; then
    echo "Exporting model to $model/merged"
    python llm_ol/experiments/llm/finetune/export_model.py \
        --checkpoint_dir $model
fi

python llm_ol/experiments/llm/finetune/inference.py \
    --test_dataset out/experiments/llm/v1/test_dataset.jsonl \
    --model $model/merged \
    --output_dir out/experiments/finetune/v3
