#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=eval
dataset=wikipedia/v2
exp_dir=out/experiments/rebel/v1/$split

# split=eval
# dataset=arxiv/v2
# exp_dir=out/experiments/rebel/v2/$split

python llm_ol/eval/hp_search.py \
    --graph $exp_dir/graph.json \
    --graph_true out/data/$dataset/train_${split}_split/test_graph.json \
    --num_samples 21 \
    --ignore_root \
    --output_dir $exp_dir
