import dataclasses
import json
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset.data_model import load_graph
from llm_ol.eval.graph_metrics import (
    edge_prec_recall_f1,
    edge_similarity,
    graph_fuzzy_match,
    motifs_wasserstein,
)
from llm_ol.experiments.post_processing import PostProcessHP, post_process
from llm_ol.utils import setup_logging
from metadata import query

FLAGS = flags.FLAGS
flags.DEFINE_string("output_file", None, "Path to the output file.", required=True)
flags.DEFINE_string("best_hp_metric", "edge_soft_f1", "Metric to use for best HP.")
flags.DEFINE_string("dataset", "wikipedia/v2", "Dataset to evaluate.")


def evaluate(G, G_true, hp):
    G, _ = post_process(G, hp)
    precision, recall, f1 = edge_prec_recall_f1(G, G_true)
    soft_precision, soft_recall, soft_f1, hard_precision, hard_recall, hard_f1 = (
        edge_similarity(G, G_true, match_threshold=0.436, skip_if_too_slow=False)
    )
    soft_graph_precision, soft_graph_recall, soft_graph_f1 = graph_fuzzy_match(
        G, G_true, direction="forward", n_iters=2
    )
    motif_wass = motifs_wasserstein(G, G_true, n=3)

    return {
        "num_nodes": nx.number_of_nodes(G),
        "num_edges": nx.number_of_edges(G),
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_soft_precision": soft_precision,
        "edge_soft_recall": soft_recall,
        "edge_soft_f1": soft_f1,
        "edge_hard_precision": hard_precision,
        "edge_hard_recall": hard_recall,
        "edge_hard_f1": hard_f1,
        "graph_soft_precision": soft_graph_precision,
        "graph_soft_recall": soft_graph_recall,
        "graph_soft_f1": soft_graph_f1,
        "motif_wass": motif_wass,
    }


def main(_):
    output_file = Path(FLAGS.output_file)
    setup_logging(output_file.parent, "test_metrics", flags=FLAGS)

    exps = [
        query(exp="memorisation", dataset=FLAGS.dataset),
        query(exp="hearst", dataset=FLAGS.dataset),
        query(exp="rebel", dataset=FLAGS.dataset),
        query(exp="prompting", k_shot=0, dataset=FLAGS.dataset),
        query(exp="prompting", k_shot=1, dataset=FLAGS.dataset),
        query(exp="prompting", k_shot=3, dataset=FLAGS.dataset),
        query(exp="finetune", reweighted=False, transfer=False, dataset=FLAGS.dataset),
        query(exp="finetune", reweighted=True, transfer=False, dataset=FLAGS.dataset),
    ]
    if FLAGS.dataset == "arxiv/v2":
        exps += [
            query(
                exp="finetune", reweighted=False, transfer=True, dataset=FLAGS.dataset
            ),
            query(
                exp="finetune", reweighted=True, transfer=True, dataset=FLAGS.dataset
            ),
        ]

    with output_file.open("w") as f:
        for exp in exps:
            logging.info("Evaluating %s", exp.name)
            G = load_graph(exp.test_output)
            G_true = load_graph(exp.test_ground_truth)
            hp = PostProcessHP(**exp.best_hp(FLAGS.best_hp_metric))
            logging.info("HP: %s", hp)
            metrics = evaluate(G, G_true, hp)
            logging.info("Metrics: %s", metrics)

            item = {"name": exp.name, "hp": dataclasses.asdict(hp), **metrics}
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    app.run(main)
