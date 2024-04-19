#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

wiki_dir=out/data/arxiv/v2

python llm_ol/dataset/train_test_split.py \
    --graph_file $wiki_dir/full/full_graph.json \
    --split_depth 1 \
    --split_prop 0.5 \
    --output_dir $wiki_dir/train_test_split

python llm_ol/dataset/train_test_split.py \
    --graph_file $wiki_dir/train_test_split/train_graph.json \
    --split_depth 1 \
    --split_prop 0.3 \
    --output_dir $wiki_dir/train_eval_split \
    --seed 2