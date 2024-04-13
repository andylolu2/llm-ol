import random

import graph_tool.all as gt
import networkx as nx
import torch
from absl import logging
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

from llm_ol.llm.embed import embed, load_embedding_model
from llm_ol.utils import batch, device, textqdm
from llm_ol.utils.nx_to_gt import nx_to_gt


def directed_diameter(G: nx.DiGraph):
    """Diameter of the largest weakly connected component of a directed graph."""
    comp = nx.weakly_connected_components(G)
    largest_comp = max(comp, key=len)
    return nx.algorithms.approximation.diameter(
        G.subgraph(largest_comp).to_undirected()
    )


def central_nodes(G: nx.DiGraph):
    G_gt, nx_to_gt_map, gt_to_nx_map = nx_to_gt(G.to_undirected())
    vertex_betweeness, edge_betweeness = gt.betweenness(G_gt)
    items = []
    for v in G_gt.vertices():
        items.append((v, vertex_betweeness[v]))

    items = sorted(items, key=lambda x: x[1], reverse=True)
    result = []
    for v, centrality in items:
        n = gt_to_nx_map[v]
        title = G.nodes[n].get("title", n)
        result.append((n, title, centrality))
    return result


def weakly_connected_component_distribution(G: nx.DiGraph):
    return [len(c) for c in nx.weakly_connected_components(G)]


def strongly_connected_component_distribution(G: nx.DiGraph):
    return [len(c) for c in nx.strongly_connected_components(G)]


def in_degree_distribution(G: nx.DiGraph):
    return [d for n, d in G.in_degree()]


def out_degree_distribution(G: nx.DiGraph):
    return [d for n, d in G.out_degree()]


def distance_distribution(G: nx.Graph):
    return list(nx.single_source_shortest_path_length(G, G.graph["root"]).values())


def random_subgraph(
    G: nx.Graph,
    radius: int = 1,
    min_size: int = 5,
    max_size: int = 30,
    undirected: bool = True,
    max_tries: int = 1000,
):
    for _ in range(max_tries):
        root = random.choice(list(G.nodes))
        G_sub = nx.ego_graph(G, root, radius, undirected=undirected)

        if min_size <= len(G_sub) <= max_size:
            return G_sub

    raise ValueError(
        f"Failed to find a subgraph with {min_size} <= size <= {max_size} after {max_tries} tries"
    )


def eigenspectrum(G: nx.Graph):
    lambda_ = nx.linalg.normalized_laplacian_spectrum(G.to_undirected())
    return lambda_.tolist()


def embed_graph(
    G: nx.Graph,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
):
    embedder, tokenizer = load_embedding_model(embedding_model)
    for nodes in batch(textqdm(G.nodes), batch_size=batch_size):
        titles = [G.nodes[n]["title"] for n in nodes]
        embedding = embed(titles, embedder, tokenizer)
        for n, e in zip(nodes, embedding):
            G.nodes[n]["embed"] = e
    return G


@torch.no_grad()
def graph_similarity(
    G1: nx.Graph,
    G2: nx.Graph,
    n_iters: int = 3,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    direction: str = "forward",
) -> float:

    def nx_to_vec(G: nx.Graph, n_iters) -> torch.Tensor:
        """Compute a graph embedding of shape (n_nodes embed_dim).

        Uses a GCN with identity weights to compute the embedding.
        """

        # Delete all node and edge attributes except for the embedding
        # Otherwise PyG might complain "Not all nodes/edges contain the same attributes"
        G = G.copy()
        for _, _, d in G.edges(data=True):
            d.clear()
        for _, d in G.nodes(data=True):
            for k in list(d.keys()):
                if k != "embed":
                    del d[k]
        pyg_G = from_networkx(G, group_node_attrs=["embed"])

        embed_dim = pyg_G.x.shape[1]
        conv = GCNConv(embed_dim, embed_dim, bias=False).to(device)
        conv.lin.weight.data = torch.eye(embed_dim, device=conv.lin.weight.device)

        pyg_batch = Batch.from_data_list([pyg_G])
        x, edge_index = pyg_batch.x, pyg_batch.edge_index  # type: ignore
        x, edge_index = x.to(device), edge_index.to(device)

        for _ in range(n_iters):
            x = conv(x, edge_index)

        return x

    if "embed" not in G1.nodes[next(iter(G1.nodes))]:
        G1 = embed_graph(G1, embedding_model=embedding_model)
    if "embed" not in G2.nodes[next(iter(G2.nodes))]:
        G2 = embed_graph(G2, embedding_model=embedding_model)

    def sim(G1, G2) -> float:
        # Compute embeddings
        x1 = nx_to_vec(G1, n_iters)
        x2 = nx_to_vec(G2, n_iters)

        # Cosine similarity matrix
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)
        sim = x1 @ x2.T

        # Aggregate similarity
        sim = (sim.amax(0).mean() + sim.amax(1).mean()) / 2
        return sim.item()

    if direction == "forward":
        return sim(G1, G2)
    elif direction == "reverse":
        return sim(nx.reverse(G1), nx.reverse(G2))
    elif direction == "undirected":
        return sim(G1.to_undirected(), G2.to_undirected())
    else:
        raise ValueError(f"Invalid direction {direction}")


def node_precision(G_pred: nx.Graph, G_true: nx.Graph):
    def title(G, n):
        return G.nodes[n].get("title").lower()

    nodes_G = {title(G_pred, n) for n in G_pred.nodes}
    nodes_G_true = {title(G_true, n) for n in G_true.nodes}
    return len(nodes_G & nodes_G_true) / len(nodes_G) if len(nodes_G) > 0 else 0


def node_recall(G_pred: nx.Graph, G_true: nx.Graph):
    def title(G, n):
        return G.nodes[n].get("title").lower()

    nodes_G = {title(G_pred, n) for n in G_pred.nodes}
    nodes_G_true = {title(G_true, n) for n in G_true.nodes}
    return (
        len(nodes_G & nodes_G_true) / len(nodes_G_true) if len(nodes_G_true) > 0 else 0
    )


def node_f1(G_pred: nx.Graph, G_true: nx.Graph):
    def title(G, n):
        return G.nodes[n].get("title").lower()

    nodes_G = {title(G_pred, n) for n in G_pred.nodes}
    nodes_G_true = {title(G_true, n) for n in G_true.nodes}
    precision = len(nodes_G & nodes_G_true) / len(nodes_G) if len(nodes_G) > 0 else 0
    recall = (
        len(nodes_G & nodes_G_true) / len(nodes_G_true) if len(nodes_G_true) > 0 else 0
    )
    return (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )


def edge_precision(G_pred: nx.Graph, G_true: nx.Graph):
    def title(G, n):
        return G.nodes[n]["title"].lower()

    edges_G = {(title(G_pred, u), title(G_pred, v)) for u, v in G_pred.edges}
    edges_G_true = {(title(G_true, u), title(G_true, v)) for u, v in G_true.edges}
    return len(edges_G & edges_G_true) / len(edges_G) if len(edges_G) > 0 else 0


def edge_recall(G_pred: nx.Graph, G_true: nx.Graph):
    def title(G, n):
        return G.nodes[n]["title"].lower()

    edges_G = {(title(G_pred, u), title(G_pred, v)) for u, v in G_pred.edges}
    edges_G_true = {(title(G_true, u), title(G_true, v)) for u, v in G_true.edges}
    return (
        len(edges_G & edges_G_true) / len(edges_G_true) if len(edges_G_true) > 0 else 0
    )


def edge_f1(G_pred: nx.Graph, G_true: nx.Graph):
    def title(G, n):
        return G.nodes[n]["title"].lower()

    edges_G = {(title(G_pred, u), title(G_pred, v)) for u, v in G_pred.edges}
    edges_G_true = {(title(G_true, u), title(G_true, v)) for u, v in G_true.edges}
    precision = len(edges_G & edges_G_true) / len(edges_G) if len(edges_G) > 0 else 0
    recall = (
        len(edges_G & edges_G_true) / len(edges_G_true) if len(edges_G_true) > 0 else 0
    )
    return (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
