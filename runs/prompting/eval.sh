#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=test
exp_dir=out/experiments/prompting/v7/$split

python llm_ol/eval/hp_search.py \
    --graph $exp_dir/graph.json \
    --graph_true out/data/wikipedia/v2/train_${split}_split/test_graph.json \
    --output_dir $exp_dir

# python llm_ol/eval/eval_single_graph.py \
#     --graph_file $exp_dir/graph.json \
#     --ground_truth_graph_file out/data/wikipedia/v2/train_test_split/test_graph.json \
#     --output_dir $exp_dir \
#     --skip_eigenspectrum \
#     --skip_central_nodes
