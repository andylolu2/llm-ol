arxiv_dir=out/data/arxiv/v1

python llm_ol/dataset/arxiv/build_pages.py \
    --output_dir $arxiv_dir/pages \
    --date_min "2020-01-01" \
    --date_max "2022-12-31" \