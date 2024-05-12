#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=eval
dir=out/experiments/hearst/svd/wiki/$split
graph_true=out/data/wikipedia/v2/train_${split}_split/test_graph.json

python llm_ol/experiments/hearst/export_graph_with_ground_truth.py \
    --extraction_dir $dir/extractions \
    --graph_true $graph_true \
    --k 5 \
    --k 10 \
    --k 15 \
    --k 20 \
    --k 25 \
    --k 50 \
    --k 100 \
    --k 150 \
    --k 200 \
    --k 250 \
    --k 300 \
    --k 500 \
    --k 1000 \
    --output_dir $dir