import random

import graph_tool.all as gt
import networkx as nx
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Batch
from torch_geometric.nn import SGConv
from torch_geometric.utils import from_networkx

from llm_ol.llm.embed import embed, load_embedding_model
from llm_ol.utils import Graph, batch, device, textqdm
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
    G: Graph,
    radius: int = 1,
    min_size: int = 5,
    max_size: int = 30,
    undirected: bool = True,
    max_tries: int = 1000,
) -> Graph:
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
    G: Graph,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> Graph:
    embedder, tokenizer = load_embedding_model(embedding_model)
    for nodes in batch(textqdm(G.nodes), batch_size=batch_size):
        titles = [G.nodes[n]["title"] for n in nodes]
        embedding = embed(titles, embedder, tokenizer)
        for n, e in zip(nodes, embedding):
            G.nodes[n]["embed"] = e
    return G


@torch.no_grad()
def graph_similarity(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    n_iters: int = 3,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    direction: str = "forward",
) -> float | None:
    if len(G1) == 0 or len(G2) == 0:
        return 0

    # Skip computation if too slow. Time complexity is O(n^2 m)
    n, m = min(len(G1), len(G2)), max(len(G1), len(G2))
    if (n**2 * m) > 10000**3:
        return None

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
        conv = SGConv(embed_dim, embed_dim, K=n_iters, bias=False).to(device)
        conv.lin.weight.data = torch.eye(embed_dim, device=conv.lin.weight.device)

        pyg_batch = Batch.from_data_list([pyg_G])
        x, edge_index = pyg_batch.x, pyg_batch.edge_index  # type: ignore
        x, edge_index = x.to(device), edge_index.to(device)
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
        sim = sim.cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(sim, maximize=True)
        score = sim[row_ind, col_ind].sum().item() / max(sim.shape)
        return score

    if direction == "forward":
        return sim(G1, G2)
    elif direction == "reverse":
        return sim(G1.reverse(copy=False), G2.reverse(copy=False))
    elif direction == "undirected":
        return sim(
            G1.to_undirected(as_view=True).to_directed(as_view=True),
            G2.to_undirected(as_view=True).to_directed(as_view=True),
        )
    else:
        raise ValueError(f"Invalid direction {direction}")


@torch.no_grad()
def edge_similarity(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    match_threshold: float = 0.9,
) -> tuple[float, float, float, float] | tuple[None, None, None, None]:
    if len(G1) == 0 or len(G2) == 0:
        return 0, 0, 0, 0

    if "embed" not in G1.nodes[next(iter(G1.nodes))]:
        G1 = embed_graph(G1, embedding_model=embedding_model)
    if "embed" not in G2.nodes[next(iter(G2.nodes))]:
        G2 = embed_graph(G2, embedding_model=embedding_model)

    def embed_edges(G, edges):
        u_emb = torch.stack([G.nodes[u]["embed"] for u, _ in edges])
        v_emb = torch.stack([G.nodes[v]["embed"] for _, v in edges])
        return u_emb, v_emb

    def edge_sim(G1, edges1, G2, edges2):
        u1_emb, v1_emb = embed_edges(G1, edges1)
        u2_emb, v2_emb = embed_edges(G2, edges2)
        u1_emb = u1_emb / u1_emb.norm(dim=-1, keepdim=True)
        v1_emb = v1_emb / v1_emb.norm(dim=-1, keepdim=True)
        u2_emb = u2_emb / u2_emb.norm(dim=-1, keepdim=True)
        v2_emb = v2_emb / v2_emb.norm(dim=-1, keepdim=True)
        sim_1 = u1_emb @ u2_emb.T
        sim_2 = v1_emb @ v2_emb.T
        return sim_1 * sim_2

    sims = []
    for edge_batch_1 in batch(G1.edges, batch_size):
        sim = [
            edge_sim(G1, edge_batch_1, G2, edge_batch_2)
            for edge_batch_2 in batch(G2.edges, batch_size)
        ]
        sims.append(torch.cat(sim, dim=-1))
    sims = torch.cat(sims, dim=0)

    # Aggregate similarity
    # Skip computation if too slow. Time complexity is O(n^2 m)
    n, m = min(sims.shape), max(sims.shape)
    if (n**2 * m) > 10000**3:
        return None, None, None, None

    row_ind, col_ind = linear_sum_assignment(sims.cpu().numpy(), maximize=True)
    avg_sim = sims[row_ind, col_ind].sum().item() / m

    matching_matrix = sims >= match_threshold
    row_ind, col_ind = linear_sum_assignment(
        matching_matrix.cpu().numpy(), maximize=True
    )
    num_matched = matching_matrix[row_ind, col_ind].sum().item()

    precision = num_matched / len(G1.edges)
    recall = num_matched / len(G2.edges)

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return avg_sim, precision, recall, f1


def node_prec_recall_f1(G_pred: nx.Graph, G_true: nx.Graph):
    if len(G_pred) == 0 or len(G_true) == 0:
        return 0, 0, 0

    def title(G, n):
        return G.nodes[n]["title"].lower()

    nodes_G = {title(G_pred, n) for n in G_pred.nodes}
    nodes_G_true = {title(G_true, n) for n in G_true.nodes}
    precision = len(nodes_G & nodes_G_true) / len(nodes_G)
    recall = len(nodes_G & nodes_G_true) / len(nodes_G_true)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def edge_prec_recall_f1(G_pred: nx.Graph, G_true: nx.Graph):
    if len(G_pred) == 0 or len(G_true) == 0:
        return 0, 0, 0

    def title(G, n):
        return G.nodes[n]["title"].lower()

    edges_G = {(title(G_pred, u), title(G_pred, v)) for u, v in G_pred.edges}
    edges_G_true = {(title(G_true, u), title(G_true, v)) for u, v in G_true.edges}
    precision = len(edges_G & edges_G_true) / len(edges_G)
    recall = len(edges_G & edges_G_true) / len(edges_G_true)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1
