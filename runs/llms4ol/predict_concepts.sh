#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=v2
# split=eval
split=test

python llm_ol/experiments/llm/llms4ol/predict_concepts.py \
    --test_dataset out/experiments/llm/$dataset/${split}_dataset.jsonl \
    --output_dir out/experiments/llms4ol/wikipedia/v1/${split}
