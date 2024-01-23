from pathlib import Path

import networkx as nx

from llm_ol.dataset import data_model

ROOT_CATEGORY_ID = "root"
ROOT_CATEGORY_NAME = "Root"

ALIASES = [
    ["math.MP", "math-ph"],
    ["stat.TH", "math.ST"],
    ["math.IT", "cs.IT"],
    ["econ.GN", "q-fin.EC"],
    ["cs.SY", "eess.SY"],
    ["cs.NA", "math.NA"],
    ["physics", "grp_physics"],
    ["econ", "grp_econ"],
    ["math", "grp_math"],
    ["q-bio", "grp_q-bio"],
    ["q-fin", "grp_q-fin"],
    ["cs", "grp_cs"],
    ["stat", "grp_stat"],
    ["eess", "grp_eess"],
]


def normalise(category_id):
    for aliases in ALIASES:
        if category_id in aliases:
            return aliases[0]
    return category_id


def load_dataset(file_path: Path, max_depth: int | None = None):
    G = data_model.load_graph(file_path)
    if max_depth is not None:
        G = nx.ego_graph(G, ROOT_CATEGORY_ID, radius=max_depth)
    return G
