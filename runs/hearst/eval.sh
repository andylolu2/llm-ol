#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

python llm_ol/eval/eval_single_graph.py \
    --graph_file out/experiments/hearst/v2/graph.json \
    --ground_truth_graph_file out/data/wikipedia/v2/train_test_split/test_graph.json \
    --output_dir out/experiments/hearst/v2
