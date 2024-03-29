#!/bin/bash
set -e

python llm_ol/eval/eval_single_graph.py \
    --graph_file out/experiments/prompting/v2/graph.json \
    --output_dir out/experiments/prompting/v2 \
    --skip_eigenspectrum
