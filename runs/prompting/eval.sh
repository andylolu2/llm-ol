#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=test
dataset=arxiv/v2
exp_dir=out/experiments/prompting/arxiv/v1/$split

python llm_ol/eval/hp_search.py \
    --graph $exp_dir/graph.json \
    --graph_true out/data/$dataset/train_${split}_split/test_graph.json \
    --output_dir $exp_dir
