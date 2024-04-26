#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/rebel/v2/eval

python llm_ol/experiments/rebel/export_graph.py \
    --input_file $exp_dir/categorised_pages.jsonl \
    --output_dir $exp_dir
