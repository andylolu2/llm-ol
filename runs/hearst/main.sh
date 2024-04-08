#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/hearst/v2/eval

python llm_ol/experiments/hearst/make_txt.py \
    --graph_file out/data/wikipedia/v2/train_eval_split/train_graph.json \
    --graph_file out/data/wikipedia/v2/train_eval_split/test_graph.json \
    --output_dir $exp_dir/abstracts

dir $exp_dir/abstracts/*.txt | sort -V > $exp_dir/abstract-list.txt
    
java \
    -classpath "corenlp/stanford-corenlp-4.5.6/*" \
    edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,pos,lemma,tokensregex \
    -tokensregex.rules llm_ol/experiments/hearst/hearst.rules \
    -fileList $exp_dir/abstract-list.txt \
    -outputDirectory $exp_dir/extractions \
    -outputFormat conll \
    -output.columns ner \
    -threads 32

# Add -noClobber if you want to avoid overwriting existing files