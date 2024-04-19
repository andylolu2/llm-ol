#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

arxiv_dir=out/data/arxiv/v2

python llm_ol/dataset/arxiv/build_categories.py \
    --output_dir $arxiv_dir/categories