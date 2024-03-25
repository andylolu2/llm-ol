#!/bin/bash

python llm_ol/experiments/finetune/build_dataset.py \
    --train_graph_file out/data/wikipedia/v2/train_test_split/train_graph.json \
    --cutoff 5 \
    --num_workers 8 \
    --output_dir out/experiments/finetune/dev