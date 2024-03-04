#!/bin/bash

python llm_ol/experiments/finetune/build_dataset.py \
    --graph_file out/data/wikipedia/v1/full/full_graph.json \
    --split_depth 2 \
    --split_prop 0.5 \
    --num_workers 16 \
    --output_dir out/experiments/finetune/v1