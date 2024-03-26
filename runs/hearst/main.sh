#!/bin/bash

python llm_ol/experiments/hearst/main.py \
    --graph_file out/data/wikipedia/v2/train_test_split/train_graph.json \
    --num_workers 8 \
    --output_dir out/experiments/hearst/v2