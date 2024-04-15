import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.eval.graph_metrics import central_nodes
from llm_ol.utils import textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "extraction_dir", None, "Directory containing the extration files", required=True
)
# flags.DEFINE_integer(
#     "weight_threshold", 1, "Edges with weights <= threshold are pruned", required=True
# )
flags.DEFINE_string("output_dir", None, "Path to the output file", required=True)


def main(_):
    extraction_dir = Path(FLAGS.extraction_dir)
    extraction_files = list(extraction_dir.glob("*.txt.conll"))
    logging.info("Loading extractions from %s", extraction_dir)

    pattern = re.compile(r"(?P<child>.*)\|\|\|(?P<parent>.*)\|\|\|(?P<rule>.*)")

    hyponyms = defaultdict(lambda: defaultdict(list))
    for extraction_file in textqdm(extraction_files):
        with open(extraction_file, "r") as f:
            for line in f:
                match = pattern.match(line)
                if match is None:
                    continue
                child = match.group("child").strip()
                parent = match.group("parent").strip()
                rule = match.group("rule").strip()
                hyponyms[parent][child].append(rule)

    # Export to a graph
    G = nx.DiGraph()
    for parent, children in hyponyms.items():
        for child, rules in children.items():
            G.add_node(parent, title=parent)
            G.add_node(child, title=child)
            G.add_edge(parent, child, weight=len(set(rules)))
    logging.info(
        "Extracted %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
    )

    # centrality = central_nodes(G)
    # G.graph["root"] = centrality[0][0]
    # logging.info("Root node: %s", G.graph["root"])
    # G.graph["root"] = None

    data_model.save_graph(G, Path(FLAGS.output_dir) / "graph.json")


if __name__ == "__main__":
    app.run(main)
