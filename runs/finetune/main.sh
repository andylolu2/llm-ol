#!/bin/bash
set -e

export HF_HUB_CACHE=out/models

accelerate launch --multi_gpu llm_ol/experiments/llm/finetune/main.py \
    --config llm_ol/experiments/llm/finetune/config.py \
    --config.train.batch_size 16 \
    --config.data.train_file out/experiments/llm/v1/train_dataset.jsonl \
    --config.data.eval_file out/experiments/llm/v1/test_dataset.jsonl \
    --config.output_dir out/experiments/finetune/v4/out
