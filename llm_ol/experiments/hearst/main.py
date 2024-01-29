import json
from collections import defaultdict
from pathlib import Path

import spacy
from absl import app, flags
from tqdm import tqdm

from llm_ol.dataset import wikipedia
from llm_ol.experiments.hearst.patterns import find_hyponyms
from llm_ol.utils.logging import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("max_depth", 1, "Maximum depth of the graph")


def extract_hyponyms(nlp, text: str):
    doc = nlp(text)

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


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir)

    # Load abstracts
    G = wikipedia.load_dataset(Path(FLAGS.graph_file), FLAGS.max_depth)
    abstracts = set()
    for _, data in G.nodes(data=True):
        for page in data["pages"]:
            abstracts.add(page["abstract"])
    del G

    # Extract hyponyms
    nlp = spacy.load("en_core_web_sm")
    hyponyms = defaultdict(lambda: defaultdict(int))
    for abstract in tqdm(abstracts):
        for src, tgt in extract_hyponyms(nlp, abstract):
            hyponyms[src][tgt] += 1

    # Save output
    with open(out_dir / "hyponyms.json") as f:
        json.dump(hyponyms, f)


if __name__ == "__main__":
    app.run(main)
