import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Collection

import spacy
from absl import app, flags

from llm_ol.dataset import data_model
from llm_ol.experiments.hearst.patterns import find_hyponyms
from llm_ol.utils import setup_logging, textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("max_depth", 1, "Maximum depth of the graph")
flags.DEFINE_integer("num_workers", os.cpu_count(), "Number of workers")


def normalize(nlp: spacy.language.Language, np_tags: Collection[str]):
    result = {}
    texts = (np_tag.replace("_", " ").replace("NP ", "") for np_tag in np_tags)
    for np_tag, doc in textqdm(
        zip(np_tags, nlp.pipe(texts, n_process=FLAGS.num_workers, batch_size=100)),
        total=len(np_tags),
    ):
        new_text = []
        for token in doc:
            if token.dep_ in ("det", "poss"):
                continue
            if token.tag_ in ("NNS", "NNPS"):
                new_text.append(token.lemma_.lower())
            else:
                new_text.append(token.text.lower())
            if token.whitespace_:
                new_text.append(token.whitespace_)
        new_text = "".join(new_text)
        result[np_tag] = new_text
    return result


def extract_hyponyms(nlp: spacy.language.Language, texts: Collection[str]):
    nlp.add_pipe("merge_noun_chunks")
    for doc in textqdm(
        nlp.pipe(texts, n_process=FLAGS.num_workers, batch_size=100), total=len(texts)
    ):
        new_text = []
        for token in doc:
            normalized = token.text if token.pos_ in ("NOUN", "PROPN") else token.lemma_
            if token.pos_ in ("NOUN", "PROPN"):
                new_text.append("NP_" + normalized.replace(" ", "_"))
            else:
                new_text.append(normalized)
            if token.whitespace_:
                new_text.append(token.whitespace_)
        new_text = "".join(new_text)
        yield from find_hyponyms(new_text)
    nlp.remove_pipe("merge_noun_chunks")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "main", flags=FLAGS)

    # Load abstracts
    G = data_model.load_graph(FLAGS.graph_file, FLAGS.max_depth)
    abstracts = set()
    for _, data in G.nodes(data=True):
        for page in data["pages"]:
            abstracts.add(page["abstract"])
    del G

    # Extract hyponyms
    nlp = spacy.load("en_core_web_sm")
    relations = list(extract_hyponyms(nlp, abstracts))

    np_tags = {src for src, _ in relations} | {tgt for _, tgt in relations}
    np_tags_to_normalized = normalize(nlp, np_tags)

    hyponyms = defaultdict(lambda: defaultdict(int))
    for src, tgt in relations:
        hyponyms[np_tags_to_normalized[src]][np_tags_to_normalized[tgt]] += 1

    # Save output
    with open(out_dir / "hyponyms.json", "w") as f:
        json.dump(hyponyms, f)


if __name__ == "__main__":
    app.run(main)
