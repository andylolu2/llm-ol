import asyncio
import json
from pathlib import Path

import aiohttp
import openai
from absl import app, flags, logging
from httpx import Limits, Timeout
from openai import AsyncOpenAI

from llm_ol.dataset import data_model
from llm_ol.experiments.finetune.templates import PROMPT_TEMPLATE
from llm_ol.utils import setup_logging, textpbar

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("max_depth", None, "Maximum depth of the graph")


async def query(client: AsyncOpenAI, title: str, abstract: str, t: float = 0) -> str:
    completion = await client.chat.completions.create(
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

    # wait for the server to start
    async with aiohttp.ClientSession() as session:
        while True:
            logging.info("Waiting for server to start")
            try:
                async with session.get("http://localhost:8080/health") as resp:
                    if resp.status == 200:
                        break
            except aiohttp.ClientConnectorError:
                await asyncio.sleep(5)
    logging.info("Server started")

    client = AsyncOpenAI(
        api_key="no-key-required",
        base_url="http://localhost:8080/v1",
        http_client=openai._base_client.AsyncHttpxClientWrapper(
            base_url="http://localhost:8080/v1",
            timeout=Timeout(None),
            limits=Limits(max_keepalive_connections=1000, max_connections=1000),
        ),
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
    sem = asyncio.Semaphore(1000)  # Limit the number of concurrent requests

    async def task(page):
        async with sem:
            try:
                out = await query(client, page["title"], page["abstract"])
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
                logging.error(
                    "Error processing page %s: %s", page["id"], repr(e) + str(e)
                )
            finally:
                pbar.update()

    await asyncio.gather(*[task(page) for page in pages])


if __name__ == "__main__":
    app.run(lambda _: asyncio.run(main(_)))
