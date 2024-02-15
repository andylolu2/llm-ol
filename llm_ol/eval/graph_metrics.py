import random

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from absl import logging
from pygraphviz import AGraph

from llm_ol.utils import sized_subplots


def compute_graph_metrics(
    G: nx.DiGraph,
    weakly_connected: bool = True,
    strongly_connected: bool = True,
    in_degree: bool = True,
    out_degree: bool = True,
    n_random_subgraphs: int = 5,
    random_subgraph_radius: int = 1,
    random_subgraph_min_size: int = 5,
    random_subgraph_max_size: int = 30,
    random_subgraph_undirected: bool = True,
):
    metrics = {
        "num_nodes": nx.number_of_nodes(G),
        "num_edges": nx.number_of_edges(G),
        "diameter": directed_diameter(G),
        "central_nodes": central_nodes(G),
    }

    all_plots = []
    if weakly_connected:
        all_plots.append(weakly_connected_component_distribution)
    if strongly_connected:
        all_plots.append(strongly_connected_component_distribution)
    if in_degree:
        all_plots.append(in_degree_distribution)
    if out_degree:
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
    return nx.diameter(G.to_undirected())


def central_nodes(G: nx.DiGraph, k: int = 5) -> list[str]:
    centrality = nx.betweenness_centrality(G.to_undirected())
    top_k = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)[:k]
    return top_k


def weakly_connected_component_distribution(G: nx.DiGraph, ax: plt.Axes):
    data = [len(c) for c in nx.weakly_connected_components(G)]
    sns.histplot(data, ax=ax)
    ax.set(
        title="Weakly Connected Component Distribution",
        xlabel="Component Size",
        ylabel="Count",
        xscale="log",
        yscale="log",
    )


def strongly_connected_component_distribution(G: nx.DiGraph, ax: plt.Axes):
    data = [len(c) for c in nx.strongly_connected_components(G)]
    sns.histplot(data, ax=ax)
    ax.set(
        title="Strongly Connected Component Distribution",
        xlabel="Component Size",
        ylabel="Count",
        xscale="log",
        yscale="log",
    )


def in_degree_distribution(G: nx.DiGraph, ax: plt.Axes):
    data = [d for n, d in G.in_degree()]
    sns.histplot(data, ax=ax)
    ax.set(
        title="In-Degree Distribution",
        xlabel="Degree",
        ylabel="Count",
        xscale="log",
        yscale="log",
    )


def out_degree_distribution(G: nx.DiGraph, ax: plt.Axes):
    data = [d for n, d in G.out_degree()]
    sns.histplot(data, ax=ax)
    ax.set(
        title="Out-Degree Distribution",
        xlabel="Degree",
        ylabel="Count",
        xscale="log",
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
