import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.eval.graph_metrics import central_nodes

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "extraction_dir", None, "Directory containing the extration files", required=True
)
flags.DEFINE_string("output_dir", None, "Path to the output file", required=True)


def main(_):
    extraction_dir = Path(FLAGS.extraction_dir)
    extraction_files = list(extraction_dir.glob("*.txt.conll"))
    logging.info("Loading extractions from %s", extraction_dir)

    pattern = re.compile(r"(?P<child>.*)\|\|\|(?P<parent>.*)\|\|\|(?P<rule>.*)")

    hyponyms = defaultdict(lambda: defaultdict(list))
    for extraction_file in extraction_files:
        with open(extraction_file, "r") as f:
            for line in f:
                match = pattern.match(line)
                if match is None:
                    continue
                child = match.group("child").strip()
                parent = match.group("parent").strip()
                rule = match.group("rule").strip()
                hyponyms[parent][child].append(rule)

    # post processing
    to_remove = set()
    for parent, children in hyponyms.items():
        for child, rules in children.items():
            # remove pairs which were not extracted by at least two distinct patterns
            if len(set(rules)) < 2:
                to_remove.add((parent, child))
            # remove any pair (y, x) if p(y, x) < p(x, y)
            if (
                child in hyponyms
                and parent in hyponyms[child]
                and len(hyponyms[child][parent]) > len(rules)
            ):
                to_remove.add((parent, child))
        # remove self loops
        if parent in children:
            to_remove.add((parent, parent))
    for parent, child in to_remove:
        del hyponyms[parent][child]

    G = nx.DiGraph()
    for parent, children in hyponyms.items():
        for child, rules in children.items():
            G.add_node(parent, title=parent)
            G.add_node(child, title=child)
            G.add_edge(parent, child, weight=len(rules))

    # G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

    centrality = central_nodes(G)
    G.graph["root"] = centrality[0][0]

    data_model.save_graph(G, Path(FLAGS.output_dir) / "graph.json")


if __name__ == "__main__":
    app.run(main)
