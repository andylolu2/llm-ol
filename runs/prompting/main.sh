python llm_ol/experiments/prompting/main.py \
    --model_file /ramdisks/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --graph_file out/data/wikipedia/v1/full/graph_depth_1.json \
    --output_dir out/experiments/prompting/dev-h-v2 \
    --num_workers 8