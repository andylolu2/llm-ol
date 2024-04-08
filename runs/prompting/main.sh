#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

python llm_ol/experiments/llm/prompting/main.py \
    --train_dataset out/experiments/llm/v2/train_dataset.jsonl \
    --test_dataset out/experiments/llm/v2/eval_dataset.jsonl \
    --k_shot 1 \
    --output_dir out/experiments/prompting/v5/eval
