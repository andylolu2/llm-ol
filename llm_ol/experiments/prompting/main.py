"""This script is idempotent."""

import json
import os
from pathlib import Path

from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.experiments.prompting.categorise_article import categorise_article
from llm_ol.experiments.prompting.create_hierarchy import create_hierarchy
from llm_ol.experiments.prompting.create_hierarchy_v2 import create_hierarchy_v2
from llm_ol.llm.cpu import load_mistral_instruct
from llm_ol.utils import setup_logging, textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("model_file", None, "Path to the model", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("max_depth", None, "Maximum depth of the graph")
flags.DEFINE_integer("num_workers", os.cpu_count(), "Number of workers")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir)
    out_file = out_dir / "categorised_pages.jsonl"

    G = data_model.load_graph(FLAGS.graph_file, FLAGS.max_depth)
    pages = {}
    for _, data in G.nodes(data=True):
        for page in data["pages"]:
            pages[page["id"]] = (page["title"], page["abstract"])
    del G

    model = load_mistral_instruct(
        FLAGS.model_file, n_threads=FLAGS.num_workers, echo=False
    )

    computed = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            computed.update({json.loads(line)["id"] for line in f})
        logging.info("Loaded %d computed pages", len(computed))

    for id, (title, abstract) in textqdm(pages.items()):
        if id in computed:
            continue

        try:
            # out = model + create_hierarchy(title, abstract)  # type: ignore
            out = model + create_hierarchy_v2(title, abstract)  # type: ignore
            with open(out_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": id,
                            "title": title,
                            "abstract": abstract,
                            "hierarchy": out["hierarchy"],
                        }
                    )
                    + "\n"
                )
        except Exception as e:
            logging.error("Error processing page %s: %s", id, repr(e) + str(e))


if __name__ == "__main__":
    app.run(main)
