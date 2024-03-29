#!/bin/bash
set -e

wiki_dir=out/data/wikipedia/v2

python llm_ol/dataset/train_test_split.py \
    --graph_file $wiki_dir/full/graph_depth_3.json \
    --split_depth 1 \
    --split_prop 0.5 \
    --output_dir $wiki_dir/train_test_split