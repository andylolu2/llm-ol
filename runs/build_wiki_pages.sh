wiki_dir=~/rds/hpc-work/llm-ol/out/data/wikipedia/v1

python llm_ol/dataset/wikipedia/build_pages.py \
    --categories_file $wiki_dir/categories/raw_categories.jsonl \
    --output_dir $wiki_dir/pages