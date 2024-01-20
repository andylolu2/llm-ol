from typing import Any, Callable

import networkx as nx
from absl import logging

from llm_ol.dataset.data_model import Category


def categories_to_graph(categories: list[Category]) -> nx.DiGraph:
    G = nx.DiGraph()
    for category in categories:
        G.add_node(category.id_, title=category.title, pages=category.pages)
        for child in category.subcategories:
            G.add_edge(category.id_, child)
    return G


def graph_to_categories(G: nx.DiGraph) -> list[Category]:
    categories = []
    for node in nx.topological_sort(G):
        children = list(G.successors(node))
        categories.append(
            Category(
                id_=node,
                title=G.nodes[node]["title"],
                subcategories=children,
                pages=G.nodes[node]["pages"],
            )
        )
    return categories


def remove_cycles(
    categories: list[Category], id_to_title: Callable[[Any], str] = lambda x: str(x)
):
    G = categories_to_graph(categories)
    while True:
        try:
            C = nx.find_cycle(G)

            for node, next_node in C:
                # Might happen if this link is present in multiple cycles
                if G.has_edge(node, next_node):
                    G.remove_edge(node, next_node)

            nodes = [id_to_title(node) for node, _ in C]
            logging.info("Found cycle: %s", " -> ".join(nodes))
        except nx.NetworkXNoCycle:
            break

    return graph_to_categories(G)


def remove_unreachable(
    categories: list[Category],
    root_category_id,
    id_to_title: Callable[[Any], str] = lambda x: str(x),
):
    G = categories_to_graph(categories)

    reachable = nx.descendants(G, root_category_id) | {root_category_id}
    unreachable = set(G.nodes) - reachable

    logging.info(
        "Removing %d unreachable nodes: %s",
        len(unreachable),
        [id_to_title(node) for node in unreachable],
    )

    G.remove_nodes_from(unreachable)

    return graph_to_categories(G)


def contract_repeated_paths(
    categories: list[Category],
    root_category_id,
    id_to_title: Callable[[Any], str] = lambda x: str(x),
):
    """Shorten all paths with repeated entries **BY NAME**.

    E.g. A -> B -> B -> C becomes A -> B -> C
    """
    id_to_category = {category.id_: category for category in categories}

    def contract(root, seen: set) -> None:
        children = id_to_category[root].subcategories.copy()
        for child in children:
            if child not in seen:
                seen.add(child)
                contract(child, seen)

            if id_to_title(root) == id_to_title(child):
                logging.info(
                    "Contracting %s -> %s (Name: %s)", child, root, id_to_title(root)
                )
                id_to_category[root].subcategories = list(
                    (
                        set(id_to_category[root].subcategories)
                        | set(id_to_category[child].subcategories)
                    )
                    - {child}
                )

    contract(root_category_id, set())

    return categories


def add_missing_leaves(
    categories: list[Category],
    id_to_title: Callable[[Any], str] = lambda x: str(x),
):
    id_to_category = {category.id_: category for category in categories}

    for category in categories:
        for child in category.subcategories:
            if child not in id_to_category:
                logging.info("Adding missing leaf: %s (%s)", id_to_title(child), child)
                id_to_category[child] = Category(id_=child, title=id_to_title(child))

    return list(id_to_category.values())


def post_process(
    categories: list[Category],
    root_category_id,
    id_to_title: Callable[[Any], str] = lambda x: str(x),
):
    raise ValueError("DEPRECATED")
    categories = add_missing_leaves(categories, id_to_title)
    categories = remove_cycles(categories, id_to_title)
    categories = remove_unreachable(categories, root_category_id, id_to_title)
    categories = contract_repeated_paths(categories, root_category_id, id_to_title)
    return categories
