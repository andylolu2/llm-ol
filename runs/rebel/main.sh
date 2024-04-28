#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=test
dataset=arxiv

python llm_ol/experiments/rebel/main.py \
    --test_dataset out/experiments/llm/arxiv/${split}_dataset.jsonl \
    --output_dir out/experiments/rebel/v2/${split}
