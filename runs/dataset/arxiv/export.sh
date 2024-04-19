#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

arxiv_dir=out/data/arxiv/v2

python llm_ol/dataset/arxiv/export_graph.py \
    --categories_file $arxiv_dir/categories/raw_categories.json \
    --pages_file $arxiv_dir/pages/papers_with_citations.jsonl \
    --min_citations 10 \
    --output_dir $arxiv_dir/full