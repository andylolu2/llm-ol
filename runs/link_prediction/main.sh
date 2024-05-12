#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

python llm_ol/experiments/link_prediction/train.py \
    --output_dir out/experiments/link_prediction/v1/train \
    --train_graph out/data/wikipedia/v2/train_eval_split/train_graph.json \
    --eval_graph out/data/wikipedia/v2/train_eval_split/test_graph.json
