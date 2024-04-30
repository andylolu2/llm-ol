# There some conflict between graph-tools and torch, need to import gt first
import graph_tool  # isort: skip

import dataclasses
import json
from itertools import product
from pathlib import Path

import numpy as np
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.eval.graph_metrics import (
    edge_prec_recall_f1,
    edge_similarity,
    embed_graph,
    graph_fuzzy_match,
    graph_similarity,
)
from llm_ol.experiments.post_processing import PostProcessHP, post_process
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph", None, "Path to the graph file.", required=True)
flags.DEFINE_string(
    "graph_true", None, "Path to the ground truth graph file.", required=True
)
flags.DEFINE_integer("num_samples", 11, "Number of thresholds to evaluate.")
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_bool("ignore_root", False, "Ignore the root node of `graph`.")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    out_file = out_dir / "hp_search.jsonl"
    setup_logging(out_dir, "hp_search", flags=FLAGS)

    G = data_model.load_graph(FLAGS.graph)
    G_true = data_model.load_graph(FLAGS.graph_true)

    if FLAGS.ignore_root:
        G.graph.pop("root", None)

    G = embed_graph(G)
    G_true = embed_graph(G_true)

    absolute_percentiles = 1 - np.geomspace(
        1 / G.number_of_edges(), 1, FLAGS.num_samples
    )
    relative_percentiles = 1 - np.geomspace(0.1, 1, FLAGS.num_samples) + 0.1

    # reverse to start with the most memory-intensive HPs
    absolute_percentiles = absolute_percentiles[::-1]
    relative_percentiles = relative_percentiles[::-1]

    if out_file.exists():
        out_file.unlink()

    for absolute_percentile, relative_percentile in product(
        absolute_percentiles, relative_percentiles
    ):
        hp = PostProcessHP(
            absolute_percentile,
            relative_percentile,
            add_root=False,
            prune_unconnected_nodes=False,
        )
        G_pruned = post_process(G, hp)

        precision, recall, f1 = edge_prec_recall_f1(G_pruned, G_true)
        soft_precision, soft_recall, soft_f1, hard_precision, hard_recall, hard_f1 = (
            edge_similarity(G_pruned, G_true, match_threshold=0.75**2)
        )
        # edge_sim, fuzzy_precision, fuzzy_recall, fuzzy_f1 = edge_similarity(
        #     G_pruned, G_true, match_threshold=0.75**2
        # )
        (
            soft_graph_precision,
            soft_graph_recall,
            soft_graph_f1,
            hard_graph_precision,
            hard_graph_recall,
            hard_graph_f1,
        ) = graph_fuzzy_match(G_pruned, G_true, threshold=0.75, direction="undirected")
        # graph_sim = graph_similarity(G_pruned, G_true)

        item = {
            # "graph_similarity": graph_sim,
            "edge_f1": f1,
            "edge_precision": precision,
            "edge_recall": recall,
            # "edge_similarity": edge_sim,
            # "fuzzy_edge_f1": fuzzy_f1,
            # "fuzzy_edge_precision": fuzzy_precision,
            # "fuzzy_edge_recall": fuzzy_recall,
            "edge_soft_precision": soft_precision,
            "edge_soft_recall": soft_recall,
            "edge_soft_f1": soft_f1,
            "edge_hard_precision": hard_precision,
            "edge_hard_recall": hard_recall,
            "edge_hard_f1": hard_f1,
            "graph_soft_precision": soft_graph_precision,
            "graph_soft_recall": soft_graph_recall,
            "graph_soft_f1": soft_graph_f1,
            "graph_hard_precision": hard_graph_precision,
            "graph_hard_recall": hard_graph_recall,
            "graph_hard_f1": hard_graph_f1,
            "hp": dataclasses.asdict(hp),
        }

        logging.info("Results: %s", json.dumps(item, indent=2))
        with out_file.open("a") as f:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    app.run(main)
