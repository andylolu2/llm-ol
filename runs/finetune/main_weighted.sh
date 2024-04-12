#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

accelerate launch --multi_gpu llm_ol/experiments/llm/finetune/training/main_weighted.py \
    --config llm_ol/experiments/llm/finetune/training/config.py \
    --config.wandb.notes "Weighted loss" \
    --config.model.name alpindale/Mistral-7B-v0.2-hf \
    --config.train.epochs 2 \
    --config.train.batch_size 8 \
    --config.data.train_file out/experiments/llm/v2/train_dataset.jsonl \
    --config.data.eval_file out/experiments/llm/v2/eval_dataset.jsonl \
    --config.output_dir out/experiments/finetune/debug
