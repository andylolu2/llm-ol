python llm_ol/experiments/finetune/build_dataset.py \
    --graph_file out/data/wikipedia/v1/full/graph_depth_3.json \
    --split_depth 2 \
    --split_prop 0.5 \
    --num_workers 4 \
    --output_dir out/experiments/finetune/depth_3