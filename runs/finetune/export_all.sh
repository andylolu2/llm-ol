#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/finetune/arxiv/v3/288/all

python llm_ol/experiments/llm/finetune/export_graph.py \
    --hierarchy_file $exp_dir/train/categorised_pages.jsonl \
    --hierarchy_file $exp_dir/eval/categorised_pages.jsonl \
    --hierarchy_file $exp_dir/test/categorised_pages.jsonl \
    --output_dir $exp_dir
