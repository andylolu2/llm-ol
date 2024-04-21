#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# python llm_ol/experiments/llm/finetune/export_model.py \
#     --checkpoint_dir out/experiments/finetune/v4/train/checkpoint-final

# accelerate launch --multi_gpu llm_ol/experiments/llm/finetune/training/main_weighted.py \
accelerate launch --multi_gpu llm_ol/experiments/llm/finetune/training/main.py \
    --config llm_ol/experiments/llm/finetune/training/config.py \
    --config.wandb.notes "Arxiv adaptation" \
    --config.model.name out/experiments/finetune/v4/train/checkpoint-final/merged \
    --config.train.epochs 3 \
    --config.train.batch_size 8 \
    --config.train.group_by_length=False \
    --config.train.lora.rank 8 \
    --config.train.lora.alpha 8 \
    --config.train.learning_rate 3e-6 \
    --config.train.warmup_steps 10 \
    --config.train.logging_steps 32 \
    --config.data.train_size 2048 \
    --config.data.eval_size 256 \
    --config.eval.eval_steps 32 \
    --config.data.train_file out/experiments/llm/arxiv/train_dataset.jsonl \
    --config.data.eval_file out/experiments/llm/arxiv/eval_dataset.jsonl \
    --config.output_dir out/experiments/finetune/arxiv/v2/train

