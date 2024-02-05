import json
from pathlib import Path
from typing import TypeAlias

import networkx as nx

ID: TypeAlias = int

# --- Data model ----
# {
#     "title": str,
#     "pages": [
#         {
#             "title": str,
#             "abstract": str,
#         }
#     ]
# }


def save_graph(G: nx.DiGraph, save_file: Path | str):
    save_file = Path(save_file)
    assert save_file.suffix == ".json"

    save_file.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G)
    with open(save_file, "w") as f:
        json.dump(data, f)


def load_graph(save_file: Path | str) -> nx.DiGraph:
    save_file = Path(save_file)
    assert save_file.suffix == ".json"

    with open(save_file, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data)
