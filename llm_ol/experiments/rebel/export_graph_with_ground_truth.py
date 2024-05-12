import json
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import spacy
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.experiments.hearst.svd_ppmi import SvdPpmiModel
from llm_ol.utils import setup_logging, textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_file", None, "Path to the inference ouptut file", required=True
)
flags.DEFINE_multi_string(
    "relations",
    ["subclass of", "instance of", "member of", "part of"],
    "List of relations to extract",
)
flags.DEFINE_string("graph_true", None, "The ground truth graph", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_multi_integer(
    "k", None, "The number of dimensions for the SVD", required=True
)
flags.DEFINE_float("threshold", 1e-5, "Threshold for edge pruning")


def parse_triplets(triplets: list[dict[str, str]]) -> set[tuple[str, str]]:
    extracted = set()
    for triplet in triplets:
        match triplet:
            case {"head": head, "type": relation, "tail": tail}:
                if relation in FLAGS.relations:
                    extracted.add((tail, head))
            case _:
                logging.error("Invalid triplet: %s", triplet)
    return extracted


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "export_graph", flags=FLAGS)

    nlp = spacy.load(
        "en_core_web_sm", enable=["tagger", "attribute_ruler", "lemmatizer"]
    )

    # Use the ground truth graph to get the true concepts
    G_true = data_model.load_graph(FLAGS.graph_true)
    true_concepts = set(G_true.nodes[n]["title"] for n in G_true.nodes)
    true_lemma_to_concepts = {}
    for doc in nlp.pipe(textqdm(true_concepts), n_process=8):
        lemma = " ".join([token.lemma_ for token in doc])
        true_lemma_to_concepts[lemma] = doc.text

    matches = []
    with open(FLAGS.input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            for parent, child in parse_triplets(item["triplets"]):
                matches.append((parent, child))
    concepts = set()
    for parent, child in matches:
        concepts.add(parent)
        concepts.add(child)

    # Map all concepts -> lemma -> canonical concept
    concept_to_lemma = {}
    for concept, doc in zip(textqdm(concepts), nlp.pipe(concepts, n_process=8)):
        lemma = " ".join([token.lemma_ for token in doc])
        concept_to_lemma[concept] = lemma

    vocab = {lemma: i + 1 for i, lemma in enumerate(set(concept_to_lemma.values()))}
    vocab["<OOV>"] = 0
    csr_m = sp.dok_matrix((len(vocab), len(vocab)), dtype=np.float64)

    for parent, child in matches:
        child_lemma = concept_to_lemma[child]
        parent_lemma = concept_to_lemma[parent]
        csr_m[vocab[parent_lemma], vocab[child_lemma]] += 1

    for k in FLAGS.k:
        model = SvdPpmiModel(csr_m, vocab, k)
        nodes = list(true_lemma_to_concepts.keys())

        # Export to a graph
        G = nx.DiGraph()
        for lemma, concept in true_lemma_to_concepts.items():
            G.add_node(lemma, title=concept)

        for u, v in textqdm(product(nodes, nodes), total=len(nodes) ** 2):
            weight = float(model.predict(u, v))
            if weight > FLAGS.threshold:
                G.add_edge(u, v, weight=weight)

        logging.info(
            "Extracted %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        G.graph["root"] = None

        data_model.save_graph(G, Path(FLAGS.output_dir) / f"k_{k}" / "graph.json")


if __name__ == "__main__":
    app.run(main)
