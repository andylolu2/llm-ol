#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=arxiv
split=test

python llm_ol/experiments/llm/prompting/main.py \
    --train_dataset out/experiments/llm/$dataset/train_dataset.jsonl \
    --test_dataset out/experiments/llm/$dataset/${split}_dataset.jsonl \
    --k_shot 0 \
    --output_dir out/experiments/prompting/arxiv/v1/${split}
