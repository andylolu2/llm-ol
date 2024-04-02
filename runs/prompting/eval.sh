#!/bin/bash
set -e

exp_dir=out/experiments/prompting/v4

python llm_ol/eval/eval_single_graph.py \
    --graph_file $exp_dir/graph.json \
    --output_dir $exp_dir \
    --skip_eigenspectrum \
    --skip_central_nodes
