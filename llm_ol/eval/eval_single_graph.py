# There some conflict between graph-tools and torch, need to import gt first
import graph_tool  # isort: skip

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
    edge_f1,
    edge_precision,
    edge_recall,
    eigenspectrum,
    graph_similarity,
    in_degree_distribution,
    node_f1,
    node_precision,
    node_recall,
    out_degree_distribution,
    random_subgraph,
    strongly_connected_component_distribution,
    weakly_connected_component_distribution,
)
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "graph_file", None, "Path to the graph files to evaluate", required=True
)
flags.DEFINE_string(
    "ground_truth_graph_file", None, "Path to the ground truth graph", required=True
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Path to the output directory to save the evaluation results",
    required=True,
)
flags.DEFINE_bool("skip_central_nodes", False, "Skip computing central nodes")
flags.DEFINE_bool("skip_eigenspectrum", False, "Skip computing eigenspectrum")
flags.DEFINE_bool("skip_similarity", False, "Skip computing similarity")
flags.DEFINE_bool("skip_random_subgraph", False, "Skip computing random subgraph")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "eval_single_graph", flags=FLAGS)

    G = data_model.load_graph(FLAGS.graph_file)
    G_true = data_model.load_graph(FLAGS.ground_truth_graph_file)
    assert isinstance(G, nx.DiGraph)
    assert isinstance(G_true, nx.DiGraph)

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
        "node_precision": lambda G: node_precision(G, G_true),
        "node_recall": lambda G: node_recall(G, G_true),
        "node_f1": lambda G: node_f1(G, G_true),
        "edge_precision": lambda G: edge_precision(G, G_true),
        "edge_recall": lambda G: edge_recall(G, G_true),
        "edge_f1": lambda G: edge_f1(G, G_true),
    }
    if not FLAGS.skip_central_nodes:
        fns["central_nodes"] = central_nodes
    if not FLAGS.skip_eigenspectrum:
        fns["eigenspectrum"] = eigenspectrum
    if not FLAGS.skip_similarity:
        fns["similarity"] = lambda G: graph_similarity(G, G_true)
        fns["similarity_rev"] = lambda G: graph_similarity(
            G, G_true, direction="reverse"
        )
        fns["similarity_sym"] = lambda G: graph_similarity(
            G, G_true, direction="undirected"
        )

    metrics = {}
    for k, fn in fns.items():
        logging.info("Computing %s", k)
        v = fn(G)
        metrics[k] = v

        if isinstance(v, Iterable) and not isinstance(v, dict):
            v = list(islice(v, 10))
        logging.info("%s: %s", k, v)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if not FLAGS.skip_random_subgraph:
        logging.info("Computing random subgraph")
        for i in range(5):
            G_sub = random_subgraph(G, 2, max_size=50)
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
