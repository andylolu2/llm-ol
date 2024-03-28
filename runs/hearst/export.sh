#!/bin/bash

dir=out/experiments/hearst/v2

python llm_ol/experiments/hearst/export_graph.py \
    --hyponyms_file $dir/hyponyms.json \
    --output_dir $dir