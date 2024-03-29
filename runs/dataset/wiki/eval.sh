#!/bin/bash
set -e

python llm_ol/eval/eval_single_graph.py \
    --graph_file out/data/wikipedia/v1/full/graph_depth_3.json \
    --output_dir out/data/wikipedia/v1/eval
