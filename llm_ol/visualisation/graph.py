from pathlib import Path

import networkx as nx
from absl import app, flags

from llm_ol.dataset import data_model

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "graph_file", None, "Path to input graph file", required=True, short_name="g"
)
flags.DEFINE_string(
    "output_dir", None, "Directory to output graph files", required=True, short_name="o"
)
flags.DEFINE_integer("depth", 2, "Depth of categories to plot", short_name="d")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = data_model.load_graph(Path(FLAGS.graph_file))
    root = list(G.nodes.keys())[0]
    layers = nx.bfs_layers(G, root)
    nodes = []
    for i, layer in enumerate(layers):
        nodes.extend(layer)
        if i == FLAGS.depth:
            break

    G = nx.subgraph(G, nodes)
    G = nx.relabel_nodes(G, {node: G.nodes[node]["title"] for node in G.nodes})

    A = nx.nx_agraph.to_agraph(G)
    A.draw(
        out_dir / "graph.pdf",
        prog="sfdp",
        args=f"-Goverlap=false -Goverlap_scaling=-5",
    )


if __name__ == "__main__":
    app.run(main)
