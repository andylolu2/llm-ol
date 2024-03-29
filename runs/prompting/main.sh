#!/bin/bash
set -e

# Start the OpenAI API server in the background
python -m vllm.entrypoints.openai.api_server \
    --port 8080 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --served-model-name gpt-3.5-turbo \
    --download-dir out/models \
    --disable-log-requests &
API_SERVER_PID=$!

# Run the main script
python llm_ol/experiments/prompting/main.py \
    --train_dataset out/experiments/llm/v1/train_dataset.jsonl \
    --test_dataset out/experiments/llm/v1/test_dataset.jsonl \
    --ports 8080 \
    --output_dir out/experiments/prompting/dev

# Kill the OpenAI API server
kill $API_SERVER_PID
