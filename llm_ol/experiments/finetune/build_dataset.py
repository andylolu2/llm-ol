import random
from itertools import islice
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)


def paths_from_root(G: nx.Graph, page: dict, n: int):
    """Find the n shortest simple paths from the root to the page.

    May return less than n paths.
    """

    # Temporarily add the page to the graph
    G.add_node(page["id"], title=page["title"])
    for category in page["categories"]:
        G.add_edge(category, page["id"])

    try:
        paths = islice(nx.shortest_simple_paths(G, G.graph["root"], page["id"]), n)
        paths = [[G.nodes[n]["title"] for n in path] for path in paths]
    finally:
        G.remove_node(page["id"])

    random.shuffle(paths)  # shuffling to avoid bias
    return paths


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "build_finetune")

    logging.info("Loading graph from %s", FLAGS.graph_file)
    G = data_model.load_graph(FLAGS.graph_file)

    # TODO: create train validation and test splits


if __name__ == "__main__":
    app.run(main)
