import json
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.dataset.utils.miscellaneous import setup_loggging

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


def main(_):
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_loggging(out_dir)

    with open(FLAGS.categories_file, "r") as f:
        categories = [json.loads(line) for line in f]
    logging.info("Total of %s non-leaf categories", len(categories))

    pages = {}
    with open(FLAGS.pages_file, "r") as f:
        for line in f:
            page = json.loads(line)
            pages[page["id"]] = page
    logging.info("Total of %s pages", len(pages))

    missing_pages = 0
    for category in categories:
        missing_pages += sum(page_id not in pages for page_id in category["pages"])
    logging.info("Missing %s pages", missing_pages)

    G = nx.DiGraph()
    for category in categories:
        pages_in_category = [
            pages[page_id] for page_id in category["pages"] if page_id in pages
        ]
        G.add_node(category["id"], title=category["title"], pages=pages_in_category)

    for category in categories:
        for subcategory in category["sub_categories"]:
            if category["id"] in G and subcategory["id"] in G:
                G.add_edge(category["id"], subcategory["id"])
    data_model.save_graph(G, out_dir / "full_graph.json")


if __name__ == "__main__":
    app.run(main)