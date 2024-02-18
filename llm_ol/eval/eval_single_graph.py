import json
from pathlib import Path

import networkx as nx
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
    setup_logging(out_dir, "eval_single_graph")

    logging.info("Loading graph from %s", FLAGS.graph_file)
    G = data_model.load_graph(FLAGS.graph_file)
    metrics, fig, G_subs = compute_graph_metrics(G)

    for k, v in metrics.items():
        logging.info("%s: %s", k, v)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig.tight_layout()
    fig.savefig(out_dir / "metrics.pdf", dpi=300)

    for i, G_sub in enumerate(G_subs):
        relabel_map = {}
        for n, data in G_sub.nodes(data=True):
            relabel_map[n] = data["title"] if "title" in data else n
        G_sub = nx.relabel_nodes(G_sub, relabel_map)
        A = nx.nx_agraph.to_agraph(G_sub)
        A.layout("fdp")
        A.draw(out_dir / f"random_subgraph_{i}.pdf")


if __name__ == "__main__":
    app.run(main)
