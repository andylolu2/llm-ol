import json
from pathlib import Path
from typing import TypeAlias

import networkx as nx
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
