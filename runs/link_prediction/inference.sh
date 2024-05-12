#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=test
exp_dir=out/experiments/link_prediction/v1

python llm_ol/experiments/link_prediction/inference.py \
    --output_dir $exp_dir/$split \
    --model_path $exp_dir/train/checkpoint-final \
    --graph_pred out/data/wikipedia/v2/train_${split}_split/test_graph.json
