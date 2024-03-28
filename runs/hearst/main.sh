#!/bin/bash

python llm_ol/experiments/hearst/make_txt.py \
    --graph_file out/data/wikipedia/v2/train_test_split/train_graph.json \
    --output_dir out/experiments/hearst/v2/abstracts

dir out/experiments/hearst/v2/abstracts/*.txt > out/experiments/hearst/v2/abstract-list.txt
    
java \
    -classpath "corenlp/stanford-corenlp-4.5.6/*" \
    edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,pos,lemma,tokensregex \
    -tokensregex.rules llm_ol/experiments/hearst/hearst.rules \
    -fileList out/experiments/hearst/v2/abstract-list.txt \
    -outputDirectory out/experiments/hearst/v2/extractions \
    -outputFormat conll \
    -output.columns ner \
    -threads 32

# Add -noClobber if you want to avoid overwriting existing files