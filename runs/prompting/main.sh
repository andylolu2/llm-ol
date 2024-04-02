#!/bin/bash

export HF_HUB_CACHE=out/models

python llm_ol/experiments/llm/prompting/main.py \
    --train_dataset out/experiments/llm/v1/train_dataset.jsonl \
    --test_dataset out/experiments/llm/v1/test_dataset.jsonl \
    --k_shot 3 \
    --output_dir out/experiments/prompting/v4

