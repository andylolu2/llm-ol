import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import aiohttp
from absl import app, flags, logging

from llm_ol.dataset.data_model import Category, save_categories
from llm_ol.dataset.post_process import post_process
from llm_ol.dataset.utils.miscellaneous import batch
from llm_ol.dataset.utils.rate_limit import Resource

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
ROOT_CATEGORY_ID = 7345184
ROOT_CATEGORY_NAME = "Main topic classifications"

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_depth", 2, "Max depth to traverse", short_name="d")
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)

wikipedia_api_limit = Resource(period=timedelta(seconds=1), limit=100)


async def api_request(session: aiohttp.ClientSession, params: dict, retries: int = 3):
    for i in range(retries):
        try:
            await wikipedia_api_limit.acquire()
            async with session.get(WIKIPEDIA_API_URL, params=params) as response:
                if response.status == 429:  # Too many requests
                    raise RuntimeError("Too many requests")
                result = await response.json()
                return result
        except Exception as e:
            if i == retries - 1:
                raise e
            else:
                logging.error("Request failed: %s. %d retries left", e, retries - i - 1)
                await asyncio.sleep(2**i)
    assert False  # Unreachable


async def get_pages_and_subcats(
    root_category_id: int, out_file: Path, max_depth: int = 0
):
    """Recursively get all pages and subcategories of a category.

    API reference: https://www.mediawiki.org/wiki/Special:MyLanguage/API:Categorymembers
    """

    # queue = asyncio.Queue()
    seen = set()
    prev_results = {}

    if out_file.exists():
        with open(out_file, "r") as f:
            for line in f:
                item = json.loads(line)
                prev_results[item["id"]] = item
    logging.info("Loaded %s seen categories", len(prev_results))

    seen.add(root_category_id)
    # await queue.put((0, root_category_id))

    async def get_category_members(category_id: int, session: aiohttp.ClientSession):
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
            result = await api_request(session, params)

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

    async def task(
        depth: int,
        category_id: int,
        session: aiohttp.ClientSession,
        task_group: asyncio.TaskGroup,
    ):
        pages, sub_categories = await get_category_members(category_id, session)

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
                task_group.create_task(
                    task(depth + 1, subcategory_id, session, task_group)
                )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session, asyncio.TaskGroup() as task_group:
        await task(0, root_category_id, session, task_group)


async def get_pages_abstract(page_ids: Iterable[int], out_file: Path):
    """Get summaries of pages.

    API reference: https://www.mediawiki.org/w/api.php?action=help&modules=query%2Bextracts
    """

    async def get_pages_summary(
        page_ids_batch: Iterable[int], session: aiohttp.ClientSession
    ):
        last_continue = {}
        while True:
            params = {
                "action": "query",
                "pageids": "|".join(map(str, page_ids_batch)),
                "prop": "extracts",
                "format": "json",
                "formatversion": "2",
                "explaintext": "true",
                "exintro": "true",
                **last_continue,
            }
            result = await api_request(session, params)
            if "error" in result:
                raise RuntimeError(result["error"])
            if "warnings" in result:
                logging.warning(result["warnings"])
            if "query" in result:
                with open(out_file, "a") as f:
                    for page in result["query"]["pages"]:
                        if page.get("missing", False):
                            continue
                        if page.get("extract", "") == "":
                            continue
                        item = {
                            "id": page["pageid"],
                            "title": page["title"],
                            "abstract": page["extract"],
                        }
                        f.write(json.dumps(item) + "\n")
            if "continue" not in result:
                return
            last_continue = result["continue"]

    prev_results = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            for line in f:
                item = json.loads(line)
                prev_results.add(item["id"])
    logging.info("Loaded %s seen pages", len(prev_results))
    page_ids = set(page_ids) - prev_results

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        tasks = []
        for page_ids_batch in batch(page_ids, 50):
            tasks.append(get_pages_summary(page_ids_batch, session))
        await asyncio.gather(*tasks)


async def async_main(_):
    # Set up
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().use_absl_log_file(log_dir=out_dir)

    asyncio.create_task(wikipedia_api_limit.replenish())

    raw_file = out_dir / "raw_results.jsonl"

    await get_pages_and_subcats(ROOT_CATEGORY_ID, raw_file, max_depth=FLAGS.max_depth)

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
    for result in results:
        category_id = result["id"]
        categories.append(
            Category(
                id_=category_id,
                title=id_to_title[category_id],
                subcategories=[page["id"] for page in result["sub_categories"]],
                pages=[page["id"] for page in result["pages"]],
            )
        )

    categories = post_process(categories, ROOT_CATEGORY_ID, lambda x: id_to_title[x])

    save_categories(categories, out_dir, format="jsonl")
    save_categories(categories, out_dir, format="owl")

    # Get summaries
    page_ids = set()
    for category in categories:
        page_ids.update(category.pages)

    logging.info("Getting summaries for %s pages", len(page_ids))

    await get_pages_abstract(page_ids, out_dir / "pages.jsonl")


def main(_):
    asyncio.run(async_main(_))


if __name__ == "__main__":
    app.run(main)
