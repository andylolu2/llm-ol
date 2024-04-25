#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# python llm_ol/experiments/llm/finetune/export_model.py \
#     --checkpoint_dir out/experiments/finetune/v4/train/checkpoint-final

base_model=out/experiments/finetune/v10/train/checkpoint-final

if [ ! -d "$base_model/merged" ]; then
    python llm_ol/experiments/llm/finetune/export_model.py \
        --checkpoint_dir $base_model
fi

# accelerate launch --multi_gpu llm_ol/experiments/llm/finetune/training/main.py \
accelerate launch --multi_gpu llm_ol/experiments/llm/finetune/training/main_weighted.py \
    --config llm_ol/experiments/llm/finetune/training/config.py \
    --config.wandb.notes "Arxiv masked adaptation" \
    --config.model.name $base_model/merged \
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
    --config.output_dir out/experiments/finetune/arxiv/v3/train

