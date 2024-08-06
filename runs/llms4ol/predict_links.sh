#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=v2
split=eval
exp_dir=out/experiments/llms4ol/wikipedia/v1
root="Main topic classifications"

python llm_ol/experiments/llm/llms4ol/predict_links.py \
    --output_dir $exp_dir/$split \
    --concepts_file $exp_dir/$split/concepts.json \
    --model_path $exp_dir/train/checkpoint-final \
    --root "$root" \
    --factor 10
