import json
from pathlib import Path

import networkx as nx
from absl import app, flags

from llm_ol.dataset import data_model

FLAGS = flags.FLAGS
flags.DEFINE_string("hyponyms_file", None, "Path to the input file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output file", required=True)


def main(_):
    with open(FLAGS.hyponyms_file, "r") as f:
        hyponyms = json.load(f)

    G = nx.DiGraph()
    for src, tgts in hyponyms.items():
        for tgt, count in tgts.items():
            if count > 1:
                G.add_node(src, title=src)
                G.add_node(tgt, title=tgt)
                G.add_edge(src, tgt, weight=count)

    # TODO: Temporary fix for the outlier "part"
    G.remove_nodes_from(["part"])
    largest_connected = max(nx.weakly_connected_components(G), key=len)
    G = nx.subgraph(G, largest_connected)

    data_model.save_graph(G, Path(FLAGS.output_dir) / "graph.json")


if __name__ == "__main__":
    app.run(main)
