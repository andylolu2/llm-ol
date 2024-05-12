#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=eval
exp_dir=out/experiments/rebel/v2/$split
graph_true=out/data/arxiv/v2/train_${split}_split/test_graph.json

python llm_ol/experiments/rebel/export_graph_with_ground_truth.py \
    --input_file $exp_dir/categorised_pages.jsonl \
    --graph_true $graph_true \
    --output_dir $exp_dir
