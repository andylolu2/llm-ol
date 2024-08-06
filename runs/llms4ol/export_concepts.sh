#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=v2
split=eval
top_k=$((2973 * 2))

# dataset=v2
# split=test
# top_k=$((8033 * 2))

python llm_ol/experiments/llm/llms4ol/export_concepts.py \
    --raw_prediction_file out/experiments/llms4ol/wikipedia/v1/${split}/categorised_pages.jsonl \
    --top_k $top_k \
    --output_dir out/experiments/llms4ol/wikipedia/v1/${split}
