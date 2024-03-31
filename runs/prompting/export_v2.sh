#!/bin/bash
set -e

python llm_ol/experiments/llm/prompting/export_graph_v2.py \
    --hierarchy_file out/experiments/prompting/v3/categorised_pages.jsonl \
    --output_dir out/experiments/prompting/v3 \
    --prune_threshold 0
