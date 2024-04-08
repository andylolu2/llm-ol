# There some conflict between graph-tools and torch, need to import gt first
import graph_tool  # isort: skip

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import networkx as nx
import numpy as np
import spacy
from absl import logging

from llm_ol.eval.graph_metrics import central_nodes, edge_f1, graph_similarity, node_f1


@dataclass
class PostProcessHP:
    weight_threshold: int = 0
    percentile_threshold: float = 1
    remove_self_loops: bool = True
    remove_inverse_edges: bool = True
    prune_unconnected_nodes: bool = True
    add_root: bool = True
    merge_nodes_by_lemma: bool = True


def post_process(G: nx.DiGraph, hp: PostProcessHP) -> nx.DiGraph:
    """Prune edges and nodes from a graph.

    Args:
        G: The input graph.
        weight_threshold: Edges with weights <= threshold are pruned.
        percentile_threshold: Outgoing edges with weight percentile > threshold are pruned.
        remove_self_loops: Remove self loops.
        remove_inverse_edges: Remove any pair (y, x) if p(y, x) < p(x, y).
        prune_unconnected_nodes: Remove nodes disconnected from the root.
            If the graph does not have a root, the largest weakly connected component is kept.
        add_root: Add a root node to the graph if it does not have one.
    """
    G = G.copy()  # type: ignore

    def weight(u, v):
        return G[u][v].get("weight", 1)

    def title(n):
        return G.nodes[n].get("title", n)

    if hp.merge_nodes_by_lemma:
        # The lemmatizer requires "tagger" and "attribute_ruler"
        nlp = spacy.load(
            "en_core_web_sm", enable=["tagger", "attribute_ruler", "lemmatizer"]
        )

        titles = list({title(n) for n in G.nodes})
        lemmas = [
            " ".join([tok.lemma_ for tok in doc])
            for doc in nlp.pipe(titles, batch_size=1000)
        ]
        title_to_lemma = {title: lemma for title, lemma in zip(titles, lemmas)}

        # Lemma to counter of labels that map to the lemma
        lemma_to_title_counts = defaultdict(Counter)
        for n in G.nodes:
            lemma_to_title_counts[title_to_lemma[title(n)]].update([title(n)])
        n_to_normalized = {
            n: lemma_to_title_counts[title_to_lemma[title(n)]].most_common(1)[0][0]
            for n in G.nodes
        }

        G_normalized = nx.DiGraph()
        if "root" in G.graph:
            G_normalized.graph["root"] = n_to_normalized[G.graph["root"]]
        for n, data in G.nodes(data=True):
            n_normalized = n_to_normalized[n]
            if not G_normalized.has_node(n_normalized) and title(n) == title(
                n_normalized
            ):
                G_normalized.add_node(n_normalized, **data)
        for u, v, data in G.edges(data=True):
            u = n_to_normalized[u]
            v = n_to_normalized[v]
            assert G_normalized.has_node(u) and G_normalized.has_node(v), (u, v)
            if G_normalized.has_edge(u, v):
                G_normalized[u][v]["weight"] += data.get("weight", 1)
            else:
                G_normalized.add_edge(u, v, **data)

        G = G_normalized

    edges_to_remove = set()
    for u, v in G.edges:
        if weight(u, v) <= hp.weight_threshold:
            edges_to_remove.add((u, v))
        if hp.remove_self_loops and u == v:
            edges_to_remove.add((u, v))
        if hp.remove_inverse_edges and G.has_edge(v, u) and weight(v, u) > weight(u, v):
            edges_to_remove.add((u, v))
    for n in G.nodes:
        edges_to_remove |= prune_edges_out_from_node(G, n, hp.percentile_threshold)
    G.remove_edges_from(edges_to_remove)

    if hp.prune_unconnected_nodes:
        if "root" in G.graph:
            root = G.graph["root"]
            connected = nx.descendants(G, root) | {root}
            G = G.subgraph(connected)
        else:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest_cc)

    if hp.add_root and "root" not in G.graph:
        centrality = central_nodes(G)
        root, _, _ = centrality[0]
        G.graph["root"] = root

    return G


def prune_edges_out_from_node(G: nx.DiGraph, node, percentile_to_keep: float):
    """Nucleus pruning: keep only the top percentile_to_keep of outgoing edges from the node."""
    assert 0 <= percentile_to_keep <= 1

    if percentile_to_keep == 1:
        return set()

    edges = list(G.out_edges(node))
    if len(edges) == 0:
        return set()

    weights = np.array([G[u][v].get("weight", 1) for u, v in edges])
    weights_sorted = np.sort(weights)[::-1]  # sort in descending order
    prune_idx = np.argmax(
        (weights_sorted / weights_sorted.sum()).cumsum() > percentile_to_keep
    )
    prune_value = weights_sorted[prune_idx]
    to_remove = {(u, v) for (u, v), w in zip(edges, weights) if w <= prune_value}
    return to_remove


def hp_search(G: nx.DiGraph, G_true: nx.DiGraph, metric: str = "edge_f1", **kwargs):
    hps = []
    keys = list(kwargs.keys())
    for values in product(*kwargs.values()):
        hps.append(PostProcessHP(**dict(zip(keys, values))))
    assert len(hps) > 0, "No hyperparameters to search over"

    if metric == "edge_f1":
        score_fn = edge_f1
    elif metric == "node_f1":
        score_fn = node_f1
    elif metric == "graph_similarity":
        score_fn = graph_similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")

    best = (None, None, -float("inf"))  # best_hp, best_G, best_score
    for hp in hps:
        G_pred = post_process(G, hp)
        score = score_fn(G_pred, G_true)
        logging.info("Score: %.5f, HP: %s", score, hp)
        if score > best[2]:
            best = (hp, G_pred, score)
    return best
