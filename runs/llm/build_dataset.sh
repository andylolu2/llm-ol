#!/bin/bash
set -e

python llm_ol/experiments/llm/build_dataset.py \
    --graph_file out/data/wikipedia/v2/train_test_split/train_graph.json \
    --cutoff 5 \
    --num_workers 8 \
    --output_file out/experiments/llm/v1/train_dataset.jsonl

python llm_ol/experiments/llm/build_dataset.py \
    --graph_file out/data/wikipedia/v2/train_test_split/test_graph.json \
    --cutoff 5 \
    --num_workers 8 \
    --output_file out/experiments/llm/v1/test_dataset.jsonl