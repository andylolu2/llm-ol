#!/bin/bash

# Start the OpenAI API server in the background
python -m vllm.entrypoints.openai.api_server \
    --download-dir out/models \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --port 8080 \
    --served-model-name gpt-3.5-turbo \
    --disable-log-requests &
API_SERVER_PID=$!

# Wait for the server to start
sleep 10

# Run the main script
python llm_ol/experiments/prompting/main.py \
    --graph_file out/data/wikipedia/v1/full/graph_depth_1.json \
    --output_dir out/experiments/prompting/dev-openai

# Kill the OpenAI API server
kill $API_SERVER_PID