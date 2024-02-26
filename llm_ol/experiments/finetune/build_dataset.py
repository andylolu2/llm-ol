from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)


def paths_to_root(G: nx.Graph, page: dict, n: int):
    for category in page["categories"]:
        G.add_edge(category, page["id"])

    try:
        paths = []
        for i, path in enumerate(
            nx.shortest_simple_paths(G, ROOT_CATEGORY_ID, page["id"])
        ):
            names = tuple(G.nodes[node]["title"] for node in path[:-1])
            paths.append(names)
            if i > n:
                break
    finally:
        G.remove_node(page["id"])

    # sort lexicographically
    return sorted(paths, key=lambda x: x)


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "build_finetune")

    logging.info("Loading graph from %s", FLAGS.graph_file)
    G = data_model.load_graph(FLAGS.graph_file)


if __name__ == "__main__":
    app.run(main)
