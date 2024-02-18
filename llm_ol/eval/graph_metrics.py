import random

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
from absl import logging
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

from llm_ol.utils import sized_subplots


def compute_graph_metrics(
    G: nx.DiGraph,
    diameter: bool = True,
    centrality: bool = True,
    n_random_subgraphs: int = 5,
    random_subgraph_radius: int = 1,
    random_subgraph_min_size: int = 5,
    random_subgraph_max_size: int = 30,
    random_subgraph_undirected: bool = True,
):
    metrics = {}
    logging.info("Computing number of nodes and edges")
    metrics["num_nodes"] = nx.number_of_nodes(G)
    metrics["num_edges"] = nx.number_of_edges(G)
    metrics["density"] = nx.density(G)
    if diameter:
        logging.info("Computing diameter")
        metrics["diameter"] = directed_diameter(G)
    if centrality:
        logging.info("Computing central nodes")
        metrics["central_nodes"] = central_nodes(G)

    all_plots = []
    all_plots.append(weakly_connected_component_distribution)
    all_plots.append(strongly_connected_component_distribution)
    all_plots.append(in_degree_distribution)
    all_plots.append(out_degree_distribution)

    fig, axs = sized_subplots(len(all_plots), n_cols=2)
    for plot_fn, ax in zip(all_plots, axs.flat):
        plot_fn(G, ax)

    G_subs: list[nx.DiGraph] = []
    for _ in range(n_random_subgraphs):
        G_sub = random_subgraph(
            G,
            random_subgraph_radius,
            random_subgraph_min_size,
            random_subgraph_max_size,
            random_subgraph_undirected,
        )
        if G_sub is not None:
            G_subs.append(G_sub)

    return metrics, fig, G_subs


def directed_diameter(G: nx.DiGraph):
    """Diameter of the largest weakly connected component of a directed graph."""
    comp = nx.weakly_connected_components(G)
    largest_comp = max(comp, key=len)
    return nx.algorithms.approximation.diameter(
        G.subgraph(largest_comp).to_undirected()
    )


def central_nodes(G: nx.DiGraph):
    centrality = nx.betweenness_centrality(G.to_undirected(), k=min(1_000, len(G)))
    items = list(centrality.items())
    items = sorted(items, key=lambda x: x[1], reverse=True)
    result = []
    for n, v in items:
        if "title" in G.nodes[n]:
            n = G.nodes[n]["title"]
        result.append((n, v))
    return result


def weakly_connected_component_distribution(G: nx.DiGraph, ax: plt.Axes):
    data = [len(c) for c in nx.weakly_connected_components(G)]
    sns.histplot(data, ax=ax)
    ax.set(
        title="Weakly Connected Component Distribution",
        xlabel="Component Size",
        ylabel="Count",
        yscale="log",
    )


def strongly_connected_component_distribution(G: nx.DiGraph, ax: plt.Axes):
    data = [len(c) for c in nx.strongly_connected_components(G)]
    sns.histplot(data, ax=ax)
    ax.set(
        title="Strongly Connected Component Distribution",
        xlabel="Component Size",
        ylabel="Count",
        yscale="log",
    )


def in_degree_distribution(G: nx.DiGraph, ax: plt.Axes):
    data = [d for n, d in G.in_degree()]
    sns.histplot(data, ax=ax)
    ax.set(
        title="In-Degree Distribution",
        xlabel="Degree",
        ylabel="Count",
        yscale="log",
    )


def out_degree_distribution(G: nx.DiGraph, ax: plt.Axes):
    data = [d for n, d in G.out_degree()]
    sns.histplot(data, ax=ax)
    ax.set(
        title="Out-Degree Distribution",
        xlabel="Degree",
        ylabel="Count",
        yscale="log",
    )


def random_subgraph(
    G: nx.DiGraph,
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
    return None


@torch.no_grad()
def graph_similarity(G1: nx.DiGraph, G2: nx.DiGraph, n_iters: int = 5) -> float:
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
