import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "hierarchy_file", None, "Path to the hierarchy directory", required=True
)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("prune_threshold", 0, "Prune weight")

pattern = re.compile(r"Main topic classifications( -> ((?!(\n|->)).)+)+")


def parse_hierarchy(hierarchy_str: str):
    paths = hierarchy_str.split("\n")
    relations = set()
    for path in paths:
        path = path.strip()
        if pattern.fullmatch(path) is None:
            logging.warn("Invalid pattern: %s", path)
            continue
        nodes = path.split(" -> ")
        for parent, child in zip(nodes[:-1], nodes[1:]):
            relations.add((parent, child))
    return relations


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "export_graph", flags=FLAGS)

    results = []
    with open(FLAGS.hierarchy_file, "r") as f:
        results = [json.loads(line) for line in f.readlines()]

    hypernyms = defaultdict(int)
    for item in results:
        try:
            for parent, child in parse_hierarchy(item["hierarchy"]):
                hypernyms[(parent, child)] += 1
        except Exception as e:
            logging.error("Error parsing hierarchy %s: %s", item["title"], e)

    logging.info("Total of %s relations", len(hypernyms))

    G = nx.DiGraph()
    G.graph["root"] = "Main topic classifications"
    for (parent, child), count in hypernyms.items():
        G.add_node(parent, title=parent)
        G.add_node(child, title=child)
        G.add_edge(parent, child, weight=count)

    # Prune graph
    edges_to_remove = set()
    for u, v, data in G.edges(data=True):
        if data["weight"] <= FLAGS.prune_threshold:
            edges_to_remove.add((u, v))
    G.remove_edges_from(edges_to_remove)
    logging.info("Removed %s edges", len(edges_to_remove))

    largest_cc = max(nx.weakly_connected_components(G), key=len)
    logging.info("Removed %s unconnected nodes", len(G) - len(largest_cc))
    G = G.subgraph(largest_cc).copy()

    data_model.save_graph(G, out_dir / "graph.json")


if __name__ == "__main__":
    app.run(main)
