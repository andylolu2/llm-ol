#!/bin/bash
set -e

export HF_HUB_CACHE=out/models

exp_dir=out/experiments/prompting/v4

python llm_ol/eval/eval_single_graph.py \
    --graph_file $exp_dir/graph.json \
    --ground_truth_graph_file out/data/wikipedia/v2/train_test_split/test_graph.json \
    --output_dir $exp_dir \
    --skip_eigenspectrum \
    --skip_central_nodes
