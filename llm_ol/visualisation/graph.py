from pathlib import Path

import networkx as nx
from absl import app, flags

from llm_ol.dataset.data_model import Category
from llm_ol.dataset.post_process import categories_to_graph

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "graph_file", None, "Path to input graph file", required=True, short_name="g"
)
flags.DEFINE_string(
    "output_dir", None, "Directory to output graph files", required=True, short_name="o"
)
flags.DEFINE_integer(
    "num_categories", 200, "Number of categories to include", short_name="n"
)
flags.DEFINE_integer(
    "scale", 5, "Scale of graph. Larger means more spread out", short_name="s"
)


def main(_):
    with open(FLAGS.graph_file, "r") as f:
        categories = [Category.model_validate_json(line) for line in f]

    G = categories_to_graph(categories)
    G = G.subgraph([category.id_ for category in categories[: FLAGS.num_categories]])
    G = nx.relabel_nodes(G, {category.id_: category.name for category in categories})

    A = nx.nx_agraph.to_agraph(G)
    A.draw(
        Path(FLAGS.output_dir, "graph.pdf"),
        prog="sfdp",
        args="-Goverlap=false -Goverlap_scaling=-5",
    )


if __name__ == "__main__":
    app.run(main)
