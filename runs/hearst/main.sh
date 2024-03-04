#!/bin/bash

python llm_ol/experiments/hearst/main.py \
    --graph_file out/data/wikipedia/v1/full/full_graph.json \
    --max_depth 3 \
    --num_workers 8 \
    --output_dir out/experiments/hearst/v1