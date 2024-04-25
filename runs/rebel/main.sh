#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=test

python llm_ol/experiments/rebel/main.py \
    --test_dataset out/experiments/llm/v2/${split}_dataset.jsonl \
    --output_dir out/experiments/rebel/v1/${split}
