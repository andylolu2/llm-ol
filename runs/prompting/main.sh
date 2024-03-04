#!/bin/bash

# Start the OpenAI API server in the background
python -m vllm.entrypoints.openai.api_server \
    --port 8080 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --served-model-name gpt-3.5-turbo \
    --download-dir out/models \
    --max-paddings 1024 \
    --disable-log-requests &
API_SERVER_PID=$!

# Wait for the API server to start
sleep 20

# Run the main script
python llm_ol/experiments/prompting/main.py \
    --graph_file out/data/wikipedia/v1/full/graph_depth_2.json \
    --output_dir out/experiments/prompting/v2

# Kill the OpenAI API server
kill $API_SERVER_PID