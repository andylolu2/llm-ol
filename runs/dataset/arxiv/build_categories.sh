#!/bin/bash

arxiv_dir=out/data/arxiv/v1

python llm_ol/dataset/arxiv/build_categories.py \
    --output_dir $arxiv_dir/categories