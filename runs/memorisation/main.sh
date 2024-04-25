#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=arxiv

python llm_ol/experiments/memorisation/export_graph.py \
    --train_dataset out/experiments/llm/$dataset/train_dataset.jsonl \
    --output_dir out/experiments/memorisation/arxiv
