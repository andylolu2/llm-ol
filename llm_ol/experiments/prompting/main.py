"""This script is idempotent."""

import asyncio
import json
from pathlib import Path

from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.experiments.prompting.create_hierarchy_v2 import create_hierarchy_v2
from llm_ol.utils import ParallelAsyncOpenAI, setup_logging, textpbar, wait_for_endpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("max_depth", None, "Maximum depth of the graph")
flags.DEFINE_integer(
    "max_concurrent_requests", 512, "Maximum number of concurrent requests per endpoint"
)
flags.DEFINE_multi_integer("ports", [], "Ports to use for the API")


async def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "categorised_pages.jsonl"

    client = ParallelAsyncOpenAI(
        base_urls=[f"http://localhost:{port}/v1" for port in FLAGS.ports],
        max_concurrent_per_client=FLAGS.max_concurrent_requests,
    )

    G = data_model.load_graph(FLAGS.graph_file, FLAGS.max_depth)
    computed = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))

    pages = []
    for _, data in G.nodes(data=True):
        for page in data["pages"]:
            if page["id"] not in computed:
                pages.append(page)

    pbar = textpbar(len(pages))

    async def task(page):
        try:
            out = await create_hierarchy_v2(client, page["title"], page["abstract"])
            with open(out_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": page["id"],
                            "title": page["title"],
                            "abstract": page["abstract"],
                            "hierarchy": out,
                        }
                    )
                    + "\n"
                )
        except Exception as e:
            logging.error("Error processing page %s: %s", page["id"], repr(e) + str(e))
        finally:
            pbar.update()

    await asyncio.gather(
        *[wait_for_endpoint(f"http://localhost:{port}/health") for port in FLAGS.ports]
    )
    await asyncio.gather(*[task(page) for page in pages])


if __name__ == "__main__":
    app.run(lambda _: asyncio.run(main(_)))
