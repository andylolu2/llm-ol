import random

import graph_tool.all as gt
import networkx as nx
import torch
from absl import logging
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

from llm_ol.utils import nx_to_gt


def directed_diameter(G: nx.DiGraph):
    """Diameter of the largest weakly connected component of a directed graph."""
    comp = nx.weakly_connected_components(G)
    largest_comp = max(comp, key=len)
    return nx.algorithms.approximation.diameter(
        G.subgraph(largest_comp).to_undirected()
    )


def central_nodes(G: nx.DiGraph):
    G_gt = nx_to_gt(G.to_undirected())
    vertex_betweeness, edge_betweeness = gt.betweenness(G_gt)
    items = []
    for v in G_gt.vertices():
        items.append((G_gt.vp["id"][v], vertex_betweeness[v]))

    items = sorted(items, key=lambda x: x[1], reverse=True)
    result = []
    for n, v in items:
        if "title" in G.nodes[n]:
            n = G.nodes[n]["title"]
        result.append((n, v))
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
        G_sub = nx.ego_graph(G, root, radius=radius, undirected=undirected)

        if min_size <= len(G_sub) <= max_size:
            return G_sub

    logging.warning(
        "Failed to find a subgraph with %d <= size <= %d after %d tries",
        min_size,
        max_size,
        max_tries,
    )


def eigenspectrum(G: nx.Graph):
    lambda_ = nx.linalg.normalized_laplacian_spectrum(G.to_undirected())
    return lambda_.tolist()


@torch.no_grad()
def graph_similarity(G1: nx.Graph, G2: nx.Graph, n_iters: int = 5) -> float:
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
        conv = GCNConv(embed_dim, embed_dim, bias=False)
        conv.lin.weight.data = torch.eye(embed_dim)

        pyg_batch = Batch.from_data_list([pyg_G])
        x, edge_index = pyg_batch.x, pyg_batch.edge_index  # type: ignore

        for _ in range(n_iters):
            x = conv(x, edge_index)

        return x

    # Compute embeddings
    x1 = nx_to_vec(G1, n_iters)
    x2 = nx_to_vec(G2, n_iters)

    # Cosine similarity matrix
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    sim = x1 @ x2.T

    # Aggregate similarity
    sim = (sim.max(0).values.mean() + sim.max(1).values.mean()) / 2
    return sim.item()
