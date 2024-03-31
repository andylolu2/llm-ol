#!/bin/bash
set -e

export HF_HUB_CACHE=out/models

accelerate launch --multi_gpu llm_ol/experiments/llm/finetune/main.py \
    --config llm_ol/experiments/llm/finetune/config.py \
    --config.model.name out/models/mistral-tiny \
    --config.train.batch_size 32 \
    --config.train.learning_rate 1e-3 \
    --config.train.epochs 0.1 \
    --config.data.train_file out/experiments/llm/v1/train_dataset.jsonl \
    --config.data.eval_file out/experiments/llm/v1/test_dataset.jsonl \
    --config.output_dir out/experiments/finetune/debug
