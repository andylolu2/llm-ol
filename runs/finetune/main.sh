#!/bin/bash

export HF_HUB_CACHE=out/models

accelerate launch --multi_gpu llm_ol/experiments/finetune/main.py \
    --config llm_ol/experiments/finetune/config.py \
    --config.data.file out/experiments/finetune/v1/train_samples.jsonl \
    --config.output_dir out/experiments/finetune/v1/out
