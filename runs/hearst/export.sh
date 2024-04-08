#!/bin/bash
set -e

dir=out/experiments/hearst/v2

python llm_ol/experiments/hearst/export_graph.py \
    --extraction_dir $dir/extractions_eval \
    --output_dir $dir