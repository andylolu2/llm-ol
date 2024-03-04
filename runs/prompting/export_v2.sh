#!/bin/bash

python llm_ol/experiments/prompting/export_graph_v2.py \
    --hierarchy_file out/experiments/prompting/dev-h-v2/categorised_pages.jsonl \
    --output_dir out/experiments/prompting/dev-h-v2
