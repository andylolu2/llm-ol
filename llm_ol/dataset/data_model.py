import json
import types
from pathlib import Path
from typing import Literal, TypeAlias

import networkx as nx
import owlready2
from pydantic import BaseModel

ID: TypeAlias = int


class Page(BaseModel):
    id_: ID
    title: str
    text: str


class Category(BaseModel):
    id_: ID
    title: str
    pages: list[ID] = []


def save_graph(G: nx.DiGraph, save_file: Path):
    assert save_file.suffix == ".json"

    save_file.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G)
    with open(save_file, "w") as f:
        json.dump(data, f)


def load_graph(save_file: Path) -> nx.DiGraph:
    assert save_file.suffix == ".json"

    with open(save_file, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def save_categories(
    categories: list[Category],
    save_dir: Path | str,
    format: Literal["jsonl", "owl"],
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        save_categories_jsonl(categories, save_dir)
    elif format == "owl":
        save_categories_owl(categories, save_dir)
    else:
        raise ValueError(f"Invalid format: {format}")


def save_categories_jsonl(categories: list[Category], save_dir: Path):
    with open(save_dir / "ontology.jsonl", "w") as f:
        for category in categories:
            f.write(category.model_dump_json() + "\n")


def load_categories_jsonl(save_dir: Path) -> list[Category]:
    categories = []
    with open(save_dir / "ontology.jsonl", "r") as f:
        for line in f:
            categories.append(Category.model_validate_json(line))
    return categories


def save_categories_owl(categories: list[Category], save_dir: Path):
    node_to_parents = {}
    for category in categories:
        for child in category.subcategories:
            node_to_parents.setdefault(child, set()).add(category.id_)

    onto = owlready2.get_ontology("http://arxiv.org/ontology/")
    with onto:
        node_to_class = {}

        def build_onto(node):
            if node in node_to_class:
                return

            parents = node_to_parents.get(node, set())
            if len(parents) == 0:
                node_class = types.new_class(str(node), (owlready2.Thing,))
            else:
                for parent in parents:
                    if parent not in node_to_class:
                        print(node, parent)
                        build_onto(parent)
                parent_classes = tuple(node_to_class[parent] for parent in parents)
                node_class = types.new_class(str(node), parent_classes)
            node_to_class[node] = node_class

        for category in categories:
            build_onto(category.id_)

    onto.save(str(save_dir / "ontology.owl"), format="rdfxml")
