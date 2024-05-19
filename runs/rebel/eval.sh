#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=eval

dataset=wikipedia/v2
exp_dir=out/experiments/rebel/svd/wiki/$split

# dataset=arxiv/v2
# exp_dir=out/experiments/rebel/svd/arxiv/$split

# for k in 5 10 15 20 25 50 100 150 200 250; do
for k in 50 100 150 200 250; do
    python llm_ol/eval/hp_search.py \
        --graph $exp_dir/k_$k/graph.json \
        --graph_true out/data/$dataset/train_${split}_split/test_graph.json \
        --num_samples 21 \
        --output_dir $exp_dir/k_$k
done