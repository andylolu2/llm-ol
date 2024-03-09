#!/bin/bash

python llm_ol/experiments/finetune/main.py \
    --config llm_ol/experiments/finetune/config.py \
    --config.model.name out/models/mistral-tiny \
    --config.data.file out/experiments/finetune/v1/train_samples.jsonl \
    --config.output_dir out/experiments/finetune/dev/out \
    --config.train.grad_acc_steps 1 \
    --config.train.batch_size 16 \
    --config.eval.batch_size 16