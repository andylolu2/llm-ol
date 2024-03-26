#!/bin/bash

# Start the OpenAI API server in the background
python -m vllm.entrypoints.openai.api_server \
    --port 8080 \
    --model out/experiments/finetune/v2/out/checkpoint-17500/merged \
    --served-model-name gpt-3.5-turbo \
    --disable-log-requests &
API_SERVER_PID=$!

# Run the main script
python llm_ol/experiments/finetune/inference.py \
    --graph_file out/data/wikipedia/v1/full/graph_depth_3.json \
    --output_dir out/experiments/finetune/v2

# Kill the OpenAI API server
kill $API_SERVER_PID
