#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dir=out/experiments/hearst/v2

python llm_ol/experiments/hearst/export_graph.py \
    --extraction_dir $dir/extractions_eval \
    --output_dir $dir