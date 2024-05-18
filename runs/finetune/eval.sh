#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=eval

# dataset=wikipedia/v2
# step=final
# exp_dir=out/experiments/finetune/v4/$step/$split
# exp_dir=out/experiments/finetune/v10/$step/$split

dataset=arxiv/v2
# step=192
# exp_dir=out/experiments/finetune/arxiv/v2/$step/$split
# step=288
# exp_dir=out/experiments/finetune/arxiv/v3/$step/$split
# step=final
# exp_dir=out/experiments/finetune/arxiv/v4/$step/$split
step=final
exp_dir=out/experiments/finetune/arxiv/v5/$step/$split

python llm_ol/eval/hp_search.py \
    --graph $exp_dir/graph.json \
    --graph_true out/data/$dataset/train_${split}_split/test_graph.json \
    --num_samples 21 \
    --output_dir $exp_dir
