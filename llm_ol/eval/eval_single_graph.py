import json
from pathlib import Path

import networkx as nx
import pygraphviz
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.eval.graph_metrics import compute_graph_metrics
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file to evaluate")
flags.DEFINE_string(
    "output_dir", None, "Path to the output directory to save the evaluation results"
)


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir)

    G = data_model.load_graph(FLAGS.graph_file)
    metrics, fig, G_subs = compute_graph_metrics(G)

    for k, v in metrics.items():
        logging.info("%s: %s", k, v)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig.tight_layout()
    fig.savefig(out_dir / "metrics.pdf", dpi=300)

    for i, G_sub in enumerate(G_subs):
        G_sub = nx.relabel_nodes(
            G_sub, {n: G_sub.nodes[n]["title"] for n in G_sub.nodes}
        )
        A = nx.nx_agraph.to_agraph(G_sub)
        A.layout("fdp")
        A.draw(out_dir / f"random_subgraph_{i}.pdf")


if __name__ == "__main__":
    app.run(main)
