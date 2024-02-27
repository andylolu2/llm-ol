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
    eigenspectrum,
    in_degree_distribution,
    out_degree_distribution,
    strongly_connected_component_distribution,
    weakly_connected_component_distribution,
)
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph files to evaluate")
flags.DEFINE_string(
    "output_dir", None, "Path to the output directory to save the evaluation results"
)
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
        "central_nodes": central_nodes,
        "weakly_connected": weakly_connected_component_distribution,
        "strongly_connected": strongly_connected_component_distribution,
        "in_degree": in_degree_distribution,
        "out_degree": out_degree_distribution,
        "eigenspectrum": eigenspectrum,
    }
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


if __name__ == "__main__":
    app.run(main)
