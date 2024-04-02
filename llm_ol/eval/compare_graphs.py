import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.eval.graph_metrics import graph_similarity
from llm_ol.llm.embed import embed, load_embedding_model
from llm_ol.utils import batch

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
flags.DEFINE_string(
    "embedding_model",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Name of the embedding model on HuggingFace",
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


def embed_graph(G: nx.DiGraph, embedder, tokenizer):
    for nodes in batch(G.nodes, FLAGS.batch_size):
        titles = [G.nodes[n]["title"] for n in nodes]
        embedding = embed(titles, embedder, tokenizer)
        for n, e in zip(nodes, embedding):
            G.nodes[n]["embed"] = e
    return G


def main(_):
    embedder, tokenizer = load_embedding_model(FLAGS.embedding_model)
    G1 = data_model.load_graph(FLAGS.graph_file_1)
    G2 = data_model.load_graph(FLAGS.graph_file_2)
    assert isinstance(G1, nx.DiGraph)
    assert isinstance(G2, nx.DiGraph)

    logging.info("Computing embeddings")
    G1 = embed_graph(G1, embedder, tokenizer)
    G2 = embed_graph(G2, embedder, tokenizer)

    logging.info("Evaluating with %s iterations", FLAGS.n_iters)
    sim = graph_similarity(G1, G2, FLAGS.n_iters)

    logging.info("Similarity: %.5f", sim)


if __name__ == "__main__":
    app.run(main)
