import json
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "hierarchy_file", None, "Path to the hierarchy directory", required=True
)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)


def parse_hierarchy(hierarchy_str: str):
    def walk_hierarchy(hierarchy):
        # Base case
        if hierarchy == "LEAF":
            return []
        elif not isinstance(hierarchy, dict) or len(hierarchy) == 0:
            raise ValueError(f"Invalid hierarchy: {hierarchy}")

        # Recursive case
        relations = []
        for parent, sub_hierarchy in hierarchy.items():
            try:
                relations += walk_hierarchy(sub_hierarchy)
            except ValueError as e:
                raise ValueError(f"Invalid hierarchy: {hierarchy}") from e
            if isinstance(sub_hierarchy, dict):
                relations += [(parent, child) for child in sub_hierarchy]
        return relations

    hierarchy = json.loads(hierarchy_str)
    return walk_hierarchy(hierarchy)


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "export_graph")

    results = []
    with open(FLAGS.hierarchy_file, "r") as f:
        results = [json.loads(line) for line in f.readlines()]

    relations = []
    for item in results:
        try:
            relations += parse_hierarchy(item["hierarchy"])
        except Exception as e:
            logging.error("Error parsing hierarchy %s: %s", item["title"], e)

    G = nx.DiGraph(relations)
    data_model.save_graph(G, out_dir / "graph.json")


if __name__ == "__main__":
    app.run(main)
