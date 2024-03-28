import asyncio
import json
from pathlib import Path

from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.experiments.finetune.templates import PROMPT_TEMPLATE
from llm_ol.utils import ParallelAsyncOpenAI, setup_logging, textpbar, wait_for_endpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer(
    "max_concurrent_requests", 512, "Maximum number of concurrent requests per endpoint"
)
flags.DEFINE_multi_integer("ports", [], "Ports to use for the API")
flags.DEFINE_integer("max_depth", None, "Maximum depth of the graph")


async def query(
    client: ParallelAsyncOpenAI, title: str, abstract: str, t: float = 0
) -> str:
    completion = await client.chat(
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.render(title=title, abstract=abstract),
            }
        ],
        model="gpt-3.5-turbo",
        temperature=t,
    )
    out = completion.choices[0].message.content
    assert isinstance(out, str)
    return out


async def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "inference", flags=FLAGS)
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
            out = await query(client, page["title"], page["abstract"], t=0.1)
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
