# There some conflict between graph-tools and torch, need to import gt first
import graph_tool  # isort: skip

import dataclasses
import json
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.eval.graph_metrics import (
    edge_f1,
    edge_precision,
    edge_recall,
    embed_graph,
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
flags.DEFINE_bool(
    "add_root", False, "Add a root node to the graph if it does not have one."
)


def main(_):
    out_dir = Path(FLAGS.output_dir)
    out_file = out_dir / "hp_search.jsonl"
    setup_logging(out_dir, "hp_search", flags=FLAGS)

    G = data_model.load_graph(FLAGS.graph)
    G_true = data_model.load_graph(FLAGS.graph_true)

    if FLAGS.ignore_root:
        G.graph.pop("root", None)

    metrics = {
        "edge_f1": edge_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "graph_similarity": partial(graph_similarity, direction="undirected"),
    }

    if "graph_similarity" in metrics:
        G = embed_graph(G)
        G_true = embed_graph(G_true)

    absolute_percentiles = 1 - np.geomspace(1 / G.number_of_edges(), 1, 11)
    relative_percentiles = 1 - np.geomspace(0.1, 1, 11) + 0.1

    computed = set()
    if out_file.exists():
        with out_file.open("r") as f:
            for line in f:
                item = json.loads(line)
                computed.add((item["absolute_percentile"], item["relative_percentile"]))

    for absolute_percentile, relative_percentile in product(
        absolute_percentiles, relative_percentiles
    ):
        if (absolute_percentile, relative_percentile) in computed:
            continue
        hp = PostProcessHP(
            absolute_percentile, relative_percentile, add_root=FLAGS.add_root
        )
        G_pruned = post_process(G, hp)
        metric_values = {name: fn(G_pruned, G_true) for name, fn in metrics.items()}
        item = {**dataclasses.asdict(hp), **metric_values}

        logging.info("Results: %s", json.dumps(item, indent=2))
        with out_file.open("a") as f:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    app.run(main)
