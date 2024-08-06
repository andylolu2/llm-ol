import json
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from absl import app, flags, logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_ol.dataset import data_model
from llm_ol.utils import batch, setup_logging, textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_string(
    "concepts_file", None, "Path to the graph to predict", required=True
)
flags.DEFINE_string("model_path", None, "Path to the model checkpoint", required=True)
flags.DEFINE_string("root", "Main topic classifications", "Root node of the graph")
flags.DEFINE_float("factor", 10, "Ratio of number of edges to keep vs nodes")


@torch.no_grad()
def main(_):
    output_dir = Path(FLAGS.output_dir)
    setup_logging(output_dir, "inference", flags=FLAGS)

    model = AutoModelForSequenceClassification.from_pretrained(
        FLAGS.model_path, num_labels=2, device_map="cuda", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_path)

    with open(FLAGS.concepts_file, "r") as f:
        node_names = list(json.load(f).keys())

    if FLAGS.root not in node_names:
        node_names.append(FLAGS.root)

    weights = []
    for uv_batch in batch(
        textqdm(product(node_names, node_names), total=len(node_names) ** 2), 2048
    ):
        us, vs = zip(*uv_batch)
        inputs = tokenizer(
            us, vs, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        output = model(**inputs)
        probs = torch.softmax(output.logits, dim=1)
        weights.append(probs[:, 1])
    weights = torch.cat(weights).reshape(len(node_names), len(node_names))
    weights = weights.float().cpu().numpy()

    # Export full graph
    G = nx.DiGraph()
    for name in node_names:
        G.add_node(name, title=name)
    for i, j in product(range(len(node_names)), repeat=2):
        G.add_edge(node_names[i], node_names[j], weight=float(weights[i, j]))
    G.graph["root"] = FLAGS.root
    logging.info(
        "Extracted %d nodes and %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    data_model.save_graph(G, Path(FLAGS.output_dir) / "graph.json")

    # Export pruned graph
    n_edges = int(len(node_names) * FLAGS.factor)
    top_idx = np.unravel_index(
        np.argpartition(weights, -n_edges, axis=None)[-n_edges:], weights.shape
    )

    G = nx.DiGraph()
    for name in node_names:
        G.add_node(name, title=name)
    for i, j in zip(*top_idx):
        G.add_edge(node_names[i], node_names[j], weight=float(weights[i, j]))
    G.graph["root"] = FLAGS.root
    logging.info(
        "Extracted %d nodes and %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    data_model.save_graph(G, Path(FLAGS.output_dir) / "graph_pruned.json")


if __name__ == "__main__":
    app.run(main)
