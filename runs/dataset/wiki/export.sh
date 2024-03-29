#!/bin/bash
set -e

wiki_dir=out/data/wikipedia/v2

python llm_ol/dataset/wikipedia/export_graph.py \
    --categories_file $wiki_dir/categories/raw_categories.jsonl \
    --pages_file $wiki_dir/pages/raw_pages.jsonl \
    --output_dir $wiki_dir/full