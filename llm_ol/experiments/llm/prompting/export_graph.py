import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "hierarchy_file", None, "Path to the hierarchy directory", required=True
)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
# flags.DEFINE_integer(
#     "weight_threshold", 0, "Edges with weights <= threshold are pruned"
# )
# flags.DEFINE_float(
#     "percentile_threshold",
#     1,
#     "Outgoing edges with weight percentile > threshold are pruned",
# )

pattern = re.compile(r"Main topic classifications( -> ((?!(\n|->)).)+)+")
empty_pattern = re.compile(r"\s*")


def parse_hierarchy(hierarchy_str: str):
    paths = hierarchy_str.split("\n")
    relations = set()
    total = 0
    num_invalid = 0
    for path in paths:
        path = path.strip()
        if empty_pattern.fullmatch(path) is not None:
            continue

        total += 1
        if pattern.fullmatch(path) is None:
            num_invalid += 1
            logging.debug("Invalid pattern: %s", path)
            continue
        nodes = path.split(" -> ")
        for parent, child in zip(nodes[:-1], nodes[1:]):
            relations.add((parent, child))
    return relations, total, num_invalid


def prune_edges(G: nx.DiGraph, node, percentile_to_keep: float):
    assert 0 <= percentile_to_keep <= 1

    edges = []
    weights = []
    for u, v, data in G.out_edges(node, data=True):
        edges.append((u, v))
        weights.append(data["weight"])
    weights = np.array(weights)
    idx = np.argsort(weights)[::-1]
    weights = weights[idx]
    edges = [edges[i] for i in idx]

    p = weights / weights.sum()
    # We do p.cumsum() - p to include the edge on boundary
    to_remove = [
        edges[i] for i in np.argwhere(p.cumsum() - p > percentile_to_keep).flatten()
    ]
    return to_remove


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "export_graph", flags=FLAGS)

    results = []
    with open(FLAGS.hierarchy_file, "r") as f:
        results = [json.loads(line) for line in f.readlines()]

    hypernyms = defaultdict(int)
    num_samples = len(results)
    num_invalid, num_paths, num_invalid_paths = 0, 0, 0
    for item in results:
        relations, total, invalid = parse_hierarchy(item["hierarchy"])
        num_paths += total
        num_invalid_paths += invalid
        num_invalid += 1 if invalid > 0 else 0
        try:
            for parent, child in relations:
                hypernyms[(parent, child)] += 1
        except Exception as e:
            logging.error("Error parsing hierarchy %s: %s", item["title"], e)

    logging.info("Total of %s samples", num_samples)
    logging.info(
        "Total of %s invalid samples (%.2f%%)",
        num_invalid,
        num_invalid / num_samples * 100,
    )
    logging.info("Total of %s paths", num_paths)
    logging.info(
        "Total of %s invalid paths (%.2f%%)",
        num_invalid_paths,
        num_invalid_paths / num_paths * 100,
    )
    logging.info("Total of %s relations", len(hypernyms))

    G = nx.DiGraph()
    G.graph["root"] = "Main topic classifications"
    for (parent, child), count in hypernyms.items():
        G.add_node(parent, title=parent)
        G.add_node(child, title=child)
        G.add_edge(parent, child, weight=count)

    # Prune graph
    # edges_to_remove = set()
    # for u, v, data in G.edges(data=True):
    #     if u == v:  # Remove self loops
    #         edges_to_remove.add((u, v))
    #     elif data["weight"] <= FLAGS.weight_threshold:
    #         edges_to_remove.add((u, v))
    # for node in G.nodes:
    #     edges_to_remove.update(prune_edges(G, node, FLAGS.percentile_threshold))
    # G.remove_edges_from(edges_to_remove)
    # logging.info("Removed %s edges", len(edges_to_remove))

    # largest_cc = max(nx.weakly_connected_components(G), key=len)
    # logging.info("Removed %s unconnected nodes", len(G) - len(largest_cc))
    # G = G.subgraph(largest_cc).copy()

    data_model.save_graph(G, out_dir / "graph.json")


if __name__ == "__main__":
    app.run(main)
