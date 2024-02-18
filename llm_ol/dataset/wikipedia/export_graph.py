import json
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.dataset.wikipedia import ROOT_CATEGORY_ID
from llm_ol.utils.logging import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "categories_file", None, "File containing categories", required=True, short_name="c"
)
flags.DEFINE_string(
    "pages_file", None, "File containing pages", required=True, short_name="p"
)
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)
flags.DEFINE_multi_integer(
    "depths", [-1, 1, 2, 3], "Depths of the graph to export", short_name="d"
)


def export_graph(pages: dict, categories: list, out_dir: Path, depth: int):
    G = nx.DiGraph()
    for category in categories:
        pages_in_category = []
        for page_id in category["pages"]:
            if page_id in pages:
                pages_in_category.append(pages[page_id])
        G.add_node(category["id"], title=category["title"], pages=pages_in_category)

    if depth >= 0:
        G = nx.ego_graph(G, ROOT_CATEGORY_ID, radius=depth)

    pages_in_G = {}
    for node, data in G.nodes(data=True):
        for page in data["pages"]:
            if page["id"] not in pages_in_G:
                pages_in_G[page["id"]] = {
                    **page,
                    "categories": [node],
                }
            else:
                pages_in_G[page["id"]]["categories"].append(node)

    data_model.save_graph(G, out_dir / f"graph_depth_{depth}.json")
    with open(out_dir / f"pages_depth_{depth}.jsonl", "w") as f:
        for page in pages_in_G.values():
            f.write(json.dumps(page) + "\n")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    with open(FLAGS.categories_file, "r") as f:
        categories = [json.loads(line) for line in f]
    logging.info("Total of %s non-leaf categories", len(categories))

    pages = {}
    with open(FLAGS.pages_file, "r") as f:
        for line in f:
            page = json.loads(line)
            pages[page["id"]] = {
                "id": page["id"],
                "title": page["title"].strip(),
                "abstract": page["abstract"].strip(),
            }
    logging.info("Total of %s pages", len(pages))

    missing_pages = 0
    for category in categories:
        missing_pages += sum(page_id not in pages for page_id in category["pages"])
    logging.info("Missing %s pages", missing_pages)

    G = nx.DiGraph()
    for category in categories:
        pages_in_category = []
        for page_id in category["pages"]:
            if page_id in pages:
                pages_in_category.append(pages[page_id])
            pages[page_id]

        G.add_node(category["id"], title=category["title"], pages=pages_in_category)

    for category in categories:
        for subcategory in category["sub_categories"]:
            if category["id"] in G and subcategory["id"] in G:
                G.add_edge(category["id"], subcategory["id"])
    data_model.save_graph(G, out_dir / "full_graph.json")

    for depth in FLAGS.depths:
        G_sub = nx.ego_graph(G, ROOT_CATEGORY_ID, radius=depth)
        data_model.save_graph(G_sub, out_dir / f"graph_depth_{depth}.json")


if __name__ == "__main__":
    app.run(main)
