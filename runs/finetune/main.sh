#!/bin/bash
set -e

export HF_HUB_CACHE=out/models

accelerate launch --multi_gpu llm_ol/experiments/finetune/main.py \
    --config llm_ol/experiments/finetune/config.py \
    --config.train.batch_size 8 \
    --config.data.file out/experiments/finetune/v3/chat_messages.jsonl \
    --config.output_dir out/experiments/finetune/v3/out
