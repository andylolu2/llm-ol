#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=arxiv/v2
cutoff=4  # use 5 for wikipedia
output_dir=out/experiments/llm/arxiv

python llm_ol/experiments/llm/build_dataset.py \
    --graph_file out/data/$dataset/train_eval_split/train_graph.json \
    --cutoff $cutoff \
    --num_workers 16 \
    --output_file $output_dir/train_dataset.jsonl

python llm_ol/experiments/llm/build_dataset.py \
    --graph_file out/data/$dataset/train_eval_split/test_graph.json \
    --cutoff $cutoff \
    --num_workers 16 \
    --output_file $output_dir/eval_dataset.jsonl

python llm_ol/experiments/llm/build_dataset.py \
    --graph_file out/data/$dataset/train_test_split/test_graph.json \
    --cutoff $cutoff \
    --num_workers 16 \
    --output_file $output_dir/test_dataset.jsonl