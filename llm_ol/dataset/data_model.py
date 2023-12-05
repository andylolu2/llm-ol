import types
from pathlib import Path
from typing import Literal

import owlready2
from pydantic import BaseModel


class Category(BaseModel):
    id_: str | int
    name: str
    children: list[str | int] = []


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


def save_categories_owl(categories: list[Category], save_dir: Path):
    node_to_parents = {}
    for category in categories:
        for child in category.children:
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
