"""This script is idempotent."""

import asyncio
import json
from pathlib import Path

from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.experiments.prompting.create_hierarchy_v2 import create_hierarchy_v2
from llm_ol.utils import setup_logging, textpbar

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("max_depth", None, "Maximum depth of the graph")


async def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir)
    out_file = out_dir / "categorised_pages.jsonl"

    G = data_model.load_graph(FLAGS.graph_file, FLAGS.max_depth)
    pages = {}
    for _, data in G.nodes(data=True):
        for page in data["pages"]:
            pages[page["id"]] = (page["title"], page["abstract"])
    del G

    computed = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))

    pbar = textpbar(len(pages) - len(computed))
    sem = asyncio.Semaphore(1000)  # Limit the number of concurrent requests

    async def task(id_, title, abstract):
        async with sem:
            try:
                out = await create_hierarchy_v2(title, abstract)
                with open(out_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": id_,
                                "title": title,
                                "abstract": abstract,
                                "hierarchy": out,
                            }
                        )
                        + "\n"
                    )
            except Exception as e:
                logging.error("Error processing page %s: %s", id_, repr(e) + str(e))
            finally:
                pbar.update()

    tasks = []
    for id_, (title, abstract) in pages.items():
        if id_ not in computed:
            tasks.append(task(id_, title, abstract))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    app.run(lambda _: asyncio.run(main(_)))
