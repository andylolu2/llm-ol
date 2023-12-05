import asyncio
import json
import traceback
from pathlib import Path
from typing import Coroutine

import aiohttp
import requests
from absl import app, flags, logging
from tqdm.asyncio import tqdm

from llm_ol.dataset.data_model import Category, save_categories
from llm_ol.dataset.post_process import (
    add_missing_leaves,
    contract_repeated_paths,
    remove_cycles,
    remove_unreachable,
)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
ROOT_CATEGORY_ID = 7345184
ROOT_CATEGORY_NAME = "Main topic classifications"

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_depth", 2, "Max depth to traverse", short_name="d")
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)
flags.DEFINE_integer("concurrency", 50, "Max number of concurrent requests")


def all_categories(last_continue=None):
    """Get all categories on Wikipedia.

    API reference: https://www.mediawiki.org/wiki/API:Allcategories
    """
    last_continue = last_continue or {}
    while True:
        logging.info("Continuing from %s", last_continue)

        params = {
            "action": "query",
            "list": "allcategories",
            "format": "json",
            "formatversion": "2",
            "acmin": 1,
            "aclimit": "max",
            **last_continue,
        }
        result = requests.get(WIKIPEDIA_API_URL, params).json()

        if "error" in result:
            raise RuntimeError(result["error"])
        if "warnings" in result:
            logging.warning(result["warnings"])
        if "query" in result:
            for page in result["query"]["pages"]:
                if "missing" in page:
                    continue
                try:
                    yield {
                        "id": page["pageid"],
                        "title": page["title"],
                        "name": page["title"].removeprefix("Category:"),
                    }
                except Exception as e:
                    logging.error("Error processing page: %s (%s)", page, repr(e))
                    raise e
        if "continue" not in result:
            break
        last_continue = result["continue"]


async def gather_with_concurrency(*coros: Coroutine, n: int = 10):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await tqdm.gather(*(sem_coro(c) for c in coros))


async def get_pages_and_subcats(
    root_category_id: int, out_file: Path, concurrency: int = 10, max_depth: int = 0
):
    queue = asyncio.Queue()
    seen = set()
    prev_results = {}

    if out_file.exists():
        with open(out_file, "r") as f:
            for line in f:
                item = json.loads(line)
                prev_results[item["id"]] = item
    logging.info("Loaded %s seen categories", len(prev_results))

    seen.add(root_category_id)
    await queue.put((0, root_category_id))

    async def worker():
        async def get_category_members(category_id: int):
            """API reference: https://www.mediawiki.org/wiki/Special:MyLanguage/API:Categorymembers"""

            if category_id in prev_results:
                return (
                    prev_results[category_id]["pages"],
                    prev_results[category_id]["sub_categories"],
                )

            pages = []
            sub_categories = []

            last_continue = {}
            for _ in range(10):  # Get at most 10x500 items
                params = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmpageid": category_id,
                    "cmtype": "page|subcat",
                    "cmprop": "ids|title|type",
                    "format": "json",
                    "formatversion": "2",
                    "cmlimit": "max",
                    **last_continue,
                }
                async with session.get(WIKIPEDIA_API_URL, params=params) as response:
                    result = await response.json()

                if "error" in result:
                    raise RuntimeError(result["error"])
                if "warnings" in result:
                    logging.warning(result["warnings"])
                if "query" in result:
                    for page in result["query"]["categorymembers"]:
                        if page.get("missing", False):
                            continue
                        if page["type"] == "page":
                            pages.append({"id": page["pageid"], "title": page["title"]})
                        elif page["type"] == "subcat":
                            sub_categories.append(
                                {
                                    "id": page["pageid"],
                                    "title": page["title"].removeprefix("Category:"),
                                }
                            )
                        else:
                            raise RuntimeError("Unknown page type: %s", page["type"])

                if "continue" not in result:
                    break
                last_continue = result["continue"]

            return pages, sub_categories

        async def process_item(item):
            depth, category_id = item

            pages, sub_categories = await get_category_members(category_id)

            if category_id not in prev_results:
                with open(out_file, "a") as f:
                    item = {
                        "id": category_id,
                        "pages": pages,
                        "sub_categories": sub_categories,
                    }
                    f.write(json.dumps(item) + "\n")

            for item in sub_categories:
                subcategory_id = item["id"]
                if subcategory_id not in seen and depth + 1 <= max_depth:
                    seen.add(subcategory_id)
                    await queue.put((depth + 1, subcategory_id))

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            while True:
                item = await queue.get()
                try:
                    await process_item(item)
                except asyncio.TimeoutError:
                    logging.error(
                        "Received timeout error on (%s), requeuing job.", item
                    )
                    await queue.put(item)
                except Exception as e:
                    trace = traceback.format_exc()
                    logging.error("Error processing item: %s\n%s", item, trace)
                    raise e
                finally:
                    queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

    # Wait until no more items in queue
    await queue.join()

    # Release workers
    for w in workers:
        w.cancel()


async def async_main(_):
    # Set up
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().use_absl_log_file(log_dir=out_dir)

    raw_file = out_dir / "raw_results.jsonl"

    await get_pages_and_subcats(
        ROOT_CATEGORY_ID,
        raw_file,
        concurrency=FLAGS.concurrency,
        max_depth=FLAGS.max_depth,
    )

    # Parse raw results
    with open(raw_file, "r") as f:
        results = [json.loads(line) for line in f]

    id_to_title = {ROOT_CATEGORY_ID: ROOT_CATEGORY_NAME}
    for result in results:
        for page in result["pages"]:
            id_to_title[page["id"]] = page["title"]
        for page in result["sub_categories"]:
            id_to_title[page["id"]] = page["title"]

    categories = []
    with open(out_dir / "pages.jsonl", "w") as f:
        for result in results:
            category_id = result["id"]
            sub_category_ids = [page["id"] for page in result["sub_categories"]]
            categories.append(
                Category(
                    id_=category_id,
                    name=id_to_title[category_id],
                    children=sub_category_ids,
                )
            )

            page = {"id": category_id, "pages": result["pages"]}
            f.write(json.dumps(page) + "\n")

    categories = add_missing_leaves(categories, lambda x: id_to_title[x])
    categories = remove_cycles(categories, lambda x: id_to_title[x])
    categories = contract_repeated_paths(
        categories, ROOT_CATEGORY_ID, lambda x: id_to_title[x]
    )
    categories = remove_unreachable(
        categories, ROOT_CATEGORY_ID, lambda x: id_to_title[x]
    )

    save_categories(categories, out_dir, format="jsonl")
    save_categories(categories, out_dir, format="owl")


def main(_):
    asyncio.run(async_main(_))


if __name__ == "__main__":
    app.run(main)
