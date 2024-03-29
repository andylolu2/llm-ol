#!/bin/bash
set -e

# Get current open file limit
OPEN_FILE_LIMIT=$(ulimit -n)
# Increase open file limit
ulimit -n 10000

# Start the OpenAI API server in the background for each GPU
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPUS=$(seq 0 $((NUM_GPUS - 1)))
API_SERVER_PIDS=()
for device_id in $GPUS; do
    CUDA_VISIBLE_DEVICES=$device_id python -m vllm.entrypoints.openai.api_server \
        --port $((8080 + $device_id)) \
        --model out/experiments/finetune/v2/out/checkpoint-17500/merged \
        --served-model-name gpt-3.5-turbo \
        --disable-log-requests &
    API_SERVER_PIDS+=($!)
done

# Run the main script
python llm_ol/experiments/finetune/inference.py \
    --graph_file out/data/wikipedia/v1/full/graph_depth_3.json \
    $(for i in $GPUS; do echo "--ports $((8080 + $i))"; done) \
    --output_dir out/experiments/finetune/v2

# Kill the OpenAI API server
for pid in ${API_SERVER_PIDS[@]}; do
    kill $pid
done

# Reset open file limit
ulimit -n $OPEN_FILE_LIMIT
