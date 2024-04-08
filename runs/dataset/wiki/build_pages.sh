#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

wiki_dir=out/data/wikipedia/v1

python llm_ol/dataset/wikipedia/build_pages.py \
    --categories_file $wiki_dir/categories/raw_categories.jsonl \
    --output_dir $wiki_dir/pages