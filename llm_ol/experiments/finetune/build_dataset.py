import json
import random
from itertools import islice
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.experiments.finetune.templates import PROMPT_TEMPLATE, RESPONSE_TEMPLATE
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)


def split_graph(
    G: nx.Graph, split_depth: int, prop: float
) -> tuple[nx.Graph, nx.Graph]:
    dist_to_root = nx.single_source_shortest_path_length(G, G.graph["root"])
    longest_dist = max(dist_to_root.values())
    nodes_to_include = {n for n, d in dist_to_root.items() if d < split_depth}
    nodes_to_split = [n for n, d in dist_to_root.items() if d == split_depth]
    nodes_1 = set(random.sample(nodes_to_split, int(prop * len(nodes_to_split))))
    logging.info(
        "Splitting graph at depth %d, selecting %d/%d nodes",
        split_depth,
        len(nodes_1),
        len(nodes_to_split),
    )

    # compute reachable nodes from nodes_1
    reachable_nodes = set()
    for n in nodes_1:
        decentants = nx.ego_graph(
            G, n, radius=longest_dist - split_depth, center=False
        ).nodes
        decentants = {n for n in decentants if dist_to_root[n] > split_depth}
        reachable_nodes.update(decentants)
    logging.info(
        "Found %d/%d (%.2f%%) nodes reachable in %d/%d hops",
        len(reachable_nodes),
        G.number_of_nodes(),
        100 * len(reachable_nodes) / G.number_of_nodes(),
        longest_dist - split_depth,
        longest_dist,
    )

    G_1 = G.subgraph(reachable_nodes | nodes_to_include | nodes_1).copy()
    G_2 = G.subgraph(set(G.nodes) - reachable_nodes - nodes_1).copy()
    logging.info(
        "Spliting graph of %d nodes into %d (%.2f%%) and %d (%.2f%%) nodes",
        G.number_of_nodes(),
        G_1.number_of_nodes(),
        100 * G_1.number_of_nodes() / G.number_of_nodes(),
        G_2.number_of_nodes(),
        100 * G_2.number_of_nodes() / G.number_of_nodes(),
    )
    logging.info(
        "Spliting graph of %d edges into %d (%.2f%%) and %d (%.2f%%) edges",
        G.number_of_edges(),
        G_1.number_of_edges(),
        100 * G_1.number_of_edges() / G.number_of_edges(),
        G_2.number_of_edges(),
        100 * G_2.number_of_edges() / G.number_of_edges(),
    )

    return G_1, G_2


def paths_from_root(G: nx.Graph, page: dict, n: int):
    """Find the n shortest simple paths from the root to the page.

    May return less than n paths.
    """

    # Temporarily add the page to the graph
    G.add_node(page["id"], title=page["title"])
    for category in page["categories"]:
        if category in G:
            G.add_edge(category, page["id"])

    try:
        paths = islice(nx.shortest_simple_paths(G, G.graph["root"], page["id"]), n)
        paths = [[G.nodes[n]["title"] for n in path] for path in paths]
    except nx.NetworkXNoPath:
        paths = []
    finally:
        G.remove_node(page["id"])

    random.shuffle(paths)  # shuffling to avoid bias
    return paths


def make_training_samples(G: nx.Graph):
    pages = {}
    for node, data in G.nodes(data=True):
        for page in data["pages"]:
            id_ = page["id"]
            if id_ not in pages:
                pages[id_] = {**page, "categories": [node]}
            else:
                pages[id_]["categories"].append(node)

    path_lengths = []
    for page in pages.values():
        paths = paths_from_root(G, page, 5)
        if len(paths) == 0:
            continue
        yield {
            "prompt": PROMPT_TEMPLATE.render(
                title=page["title"], abstract=page["abstract"]
            ),
            "response": RESPONSE_TEMPLATE.render(paths=paths),
        }
        path_lengths.append(len(paths))

    logging.info("Number of samples: %d", len(path_lengths))
    logging.info("Average path length: %.2f", sum(path_lengths) / len(path_lengths))


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "build_finetune")

    G = data_model.load_graph(FLAGS.graph_file)
    G_test, G_train = split_graph(G, split_depth=1, prop=0.1)

    with open(out_dir / "train_samples.jsonl", "w") as f:
        for chat in make_training_samples(G_train):
            f.write(json.dumps(chat) + "\n")

    with open(out_dir / "test_samples.jsonl", "w") as f:
        for chat in make_training_samples(G_test):
            f.write(json.dumps(chat) + "\n")


if __name__ == "__main__":
    app.run(main)
