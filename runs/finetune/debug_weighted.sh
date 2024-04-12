#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

python llm_ol/experiments/llm/finetune/training/main_weighted.py \
    --config llm_ol/experiments/llm/finetune/training/config.py \
    --config.model.name out/models/mistral-tiny \
    --config.train.batch_size 32 \
    --config.train.learning_rate 1e-3 \
    --config.train.epochs 0.1 \
    --config.data.train_file out/experiments/llm/v2/train_dataset.jsonl \
    --config.data.eval_file out/experiments/llm/v2/eval_dataset.jsonl \
    --config.output_dir out/experiments/finetune/debug
