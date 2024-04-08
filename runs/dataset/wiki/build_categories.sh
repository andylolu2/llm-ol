#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

python llm_ol/dataset/wikipedia/build_categories.py \
    --max_depth 4 \
    --output_dir out/data/wikipedia/v1/categories