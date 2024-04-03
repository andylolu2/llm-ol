import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.eval.graph_metrics import graph_similarity

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "graph_file_1",
    None,
    "Path to the first graph file",
    required=True,
)
flags.DEFINE_string(
    "graph_file_2",
    None,
    "Path to the second graph file",
    required=True,
)
flags.DEFINE_integer(
    "n_iters",
    3,
    "Number of iterations for the similarity computation",
)
flags.DEFINE_integer(
    "batch_size",
    64,
    "Batch size for the embedding model",
)


def main(_):
    G1 = data_model.load_graph(FLAGS.graph_file_1)
    G2 = data_model.load_graph(FLAGS.graph_file_2)
    assert isinstance(G1, nx.DiGraph)
    assert isinstance(G2, nx.DiGraph)

    logging.info("Evaluating with %s iterations", FLAGS.n_iters)
    sim = graph_similarity(G1, G2, FLAGS.n_iters)

    logging.info("Similarity: %.5f", sim)


if __name__ == "__main__":
    app.run(main)
