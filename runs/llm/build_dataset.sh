#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

python llm_ol/experiments/llm/build_dataset.py \
    --graph_file out/data/wikipedia/v2/train_eval_split/train_graph.json \
    --cutoff 5 \
    --num_workers 16 \
    --output_file out/experiments/llm/v2/train_dataset.jsonl

python llm_ol/experiments/llm/build_dataset.py \
    --graph_file out/data/wikipedia/v2/train_eval_split/test_graph.json \
    --cutoff 5 \
    --num_workers 16 \
    --output_file out/experiments/llm/v2/eval_dataset.jsonl

python llm_ol/experiments/llm/build_dataset.py \
    --graph_file out/data/wikipedia/v2/train_test_split/test_graph.json \
    --cutoff 5 \
    --num_workers 16 \
    --output_file out/experiments/llm/v2/test_dataset.jsonl