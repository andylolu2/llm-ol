arxiv_dir=~/rds/hpc-work/llm-ol/out/data/arxiv/v1

python llm_ol/dataset/arxiv/build_categories.py \
    --output_dir $arxiv_dir/categories