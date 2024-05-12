#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=eval

# exp_dir=out/experiments/rebel/svd/wiki/$split
# graph_true=out/data/wikipedia/v2/train_${split}_split/test_graph.json
# threshold=1e-5

exp_dir=out/experiments/rebel/svd/arxiv/$split
graph_true=out/data/arxiv/v2/train_${split}_split/test_graph.json
threshold=0

python llm_ol/experiments/rebel/export_graph_with_ground_truth.py \
    --input_file $exp_dir/categorised_pages.jsonl \
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
    --threshold $threshold \
    --output_dir $exp_dir

# split=test

# exp_dir=out/experiments/rebel/svd/wiki/$split
# graph_true=out/data/wikipedia/v2/train_${split}_split/test_graph.json
# threshold=1e-5
# k=50

# exp_dir=out/experiments/rebel/svd/arxiv/$split
# graph_true=out/data/arxiv/v2/train_${split}_split/test_graph.json
# threshold=0
# k=200

# python llm_ol/experiments/rebel/export_graph_with_ground_truth.py \
#     --input_file $exp_dir/categorised_pages.jsonl \
#     --graph_true $graph_true \
#     --k $k \
#     --threshold $threshold \
#     --output_dir $exp_dir