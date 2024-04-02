#!/bin/bash
set -e

exp_dir=out/experiments/finetune/v3

python llm_ol/experiments/llm/finetune/export_graph.py \
    --hierarchy_file $exp_dir/categorised_pages.jsonl \
    --output_dir $exp_dir \
    --prune_threshold 0
