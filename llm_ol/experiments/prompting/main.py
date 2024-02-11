"""This script is idempotent."""

import json
import os
from pathlib import Path

from absl import app, flags, logging

from llm_ol.dataset import wikipedia
from llm_ol.llm.cpu import load_mistral_instruct
from llm_ol.llm.templates import categorise_article
from llm_ol.utils.logging import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("model_file", None, "Path to the model", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("max_depth", 1, "Maximum depth of the graph")
flags.DEFINE_integer("num_workers", os.cpu_count(), "Number of workers")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir)
    out_file = out_dir / "categoried_pages.jsonl"

    G = wikipedia.load_dataset(Path(FLAGS.graph_file), FLAGS.max_depth)
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
        logging.info("Loaded %s computed pages", len(computed))

    for id, (title, abstract) in pages.items():
        if id in computed:
            continue

        out = model + categorise_article(title, abstract, categories=[])
        with open(out_dir / "categoried_pages.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": id,
                        "title": title,
                        "abstract": abstract,
                        "categories": out["cats"],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    app.run(main)
