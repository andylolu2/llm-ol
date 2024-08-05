#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=wikipedia/v2
# dataset=arxiv/v2

# replace / with _
dataset_name=$(echo $dataset | sed 's/\//_/g')

python llm_ol/eval/test_metrics.py \
    --output_file out/eval/$dataset_name/test_metrics_w_link_pred.jsonl \
    --dataset $dataset
