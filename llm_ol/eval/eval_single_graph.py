import json
from itertools import islice
from pathlib import Path
from typing import Iterable

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.eval.graph_metrics import (
    central_nodes,
    directed_diameter,
    distance_distribution,
    eigenspectrum,
    in_degree_distribution,
    out_degree_distribution,
    random_subgraph,
    strongly_connected_component_distribution,
    weakly_connected_component_distribution,
)
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph files to evaluate")
flags.DEFINE_string(
    "output_dir", None, "Path to the output directory to save the evaluation results"
)
flags.DEFINE_bool("skip_central_nodes", False, "Skip computing central nodes")
flags.DEFINE_bool("skip_eigenspectrum", False, "Skip computing eigenspectrum")
flags.mark_flags_as_required(["graph_file", "output_dir"])


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "eval_single_graph")

    G = data_model.load_graph(FLAGS.graph_file)
    assert isinstance(G, nx.DiGraph)

    fns = {
        "num_nodes": nx.number_of_nodes,
        "num_edges": nx.number_of_edges,
        "density": nx.density,
        "diameter": directed_diameter,
        "weakly_connected": weakly_connected_component_distribution,
        "strongly_connected": strongly_connected_component_distribution,
        "in_degree": in_degree_distribution,
        "out_degree": out_degree_distribution,
        "distance": distance_distribution,
    }
    if not FLAGS.skip_central_nodes:
        fns["central_nodes"] = central_nodes
    if not FLAGS.skip_eigenspectrum:
        fns["eigenspectrum"] = eigenspectrum

    metrics = {}
    for k, fn in fns.items():
        logging.info("Computing %s", k)
        v = fn(G)
        metrics[k] = v

        if isinstance(v, Iterable):
            v = list(islice(v, 10))
        logging.info("%s: %s", k, v)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Computing random subgraph")
    for i in range(5):
        G_sub = random_subgraph(G, 2)
        if G_sub is None:
            logging.warn("Could not compute random subgraph")
            break
        relabel_map = {}
        for n, data in G_sub.nodes(data=True):
            relabel_map[n] = data["title"] if "title" in data else n
        G_sub = nx.relabel_nodes(G_sub, relabel_map)
        A = nx.nx_agraph.to_agraph(G_sub)
        A.layout("fdp")
        A.draw(out_dir / f"random_subgraph_{i}.pdf")


if __name__ == "__main__":
    app.run(main)
