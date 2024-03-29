#!/bin/bash
set -e

python llm_ol/dataset/wikipedia/build_categories.py \
    --max_depth 4 \
    --output_dir out/data/wikipedia/v1/categories