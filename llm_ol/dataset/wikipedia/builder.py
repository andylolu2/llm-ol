import asyncio
from collections import deque
from pathlib import Path
from shutil import rmtree

import aiohttp
from absl import app, flags, logging
from bs4 import BeautifulSoup

from llm_ol.dataset.data_model import Category, save_categories

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
ROOT_CATEGORY_ID = "Main_topic_classifications"
ROOT_CATEGORY_NAME = "Main topic classifications"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)
flags.DEFINE_integer("depth", 2, "Depth of categories to fetch", short_name="d")
flags.DEFINE_float("rate", 100, "Max number of requests per second", short_name="r")
flags.DEFINE_integer("retries", 3, "Number of retries per request")


def make_url(base_url: str, params: dict[str, str]) -> str:
    params_str = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{base_url}?{params_str}"


async def make_requests(
    urls: list[str], rate: float | None = None, retries: int = 1
) -> list[dict]:
    """Make requests to the given URLs in parallel.

    Args:
        urls: List of URLs to make requests to.
        rate: Max number of requests per second. If None, no rate limit is applied.

    Returns:
        List of JSON responses.
    """
    async with aiohttp.ClientSession() as session:

        async def task(url: str, delay: float = 0):
            attempts = 0
            while attempts < retries:
                try:
                    await asyncio.sleep(delay)
                    logging.debug(f"Making request to {url}")
                    async with session.get(url) as response:
                        return await response.json()
                except Exception as e:
                    logging.warning(
                        "Retry %d: Error making request to %s: %s", attempts, url, e
                    )
                    attempts += 1
                    delay *= 2

            logging.error(
                "Failed to make request to %s after %d attempts", url, retries
            )

        time_per_request = 0 if rate is None else 1 / rate
        tasks = [task(url, i * time_per_request) for i, url in enumerate(urls)]
        return await asyncio.gather(*tasks)


def process_result(
    category_id: str,
    category_name: str,
    result: dict,
) -> tuple[Category, list[str], list[str]]:
    # Sample request
    # https://en.wikipedia.org/w/api.php?action=categorytree&format=json&category=Category:Contents&options={"depth":2}

    html = result["categorytree"]["*"]
    soup = BeautifulSoup(html, "html.parser")

    """
    Sample html
    <div class="CategoryTreeSection">
        <div class="CategoryTreeItem">
            <span class="CategoryTreeBullet">
                <span class="CategoryTreeToggle" data-ct-title="Academic_disciplines" data-ct-loaded="1" data-ct-state="expanded"></span> 
            </span>
            <a href="/wiki/Category:Academic_disciplines" title="Category:Academic disciplines">Academic disciplines</a>
        </div>
        <div class="CategoryTreeChildren">
            <div class="CategoryTreeSection">
            <div class="CategoryTreeItem">
                <span class="CategoryTreeBullet">
                    <span class="CategoryTreeToggle" data-ct-title="Subfields_by_academic_discipline" data-ct-state="collapsed"></span>
                </span>
                <a href="/wiki/Category:Subfields_by_academic_discipline" title="Category:Subfields by academic discipline">Subfields by academic discipline</a>
            </div>
            <div class="CategoryTreeChildren" style="display:none"></div>
        </div>
    </div>
    """

    children_ids = []
    children_names = []
    # Find all top-level `CategoryTreeSection`s
    for category_tree_section in soup.find_all(
        "div", class_="CategoryTreeSection", recursive=False
    ):
        # Find all `CategoryTreeItem`s
        for category_tree_item in category_tree_section.find_all(
            "div", class_="CategoryTreeItem", recursive=False
        ):
            # Get the <a> tag
            category_tree_link = category_tree_item.find("a")
            id_ = category_tree_link["href"].removeprefix("/wiki/Category:")
            name = category_tree_link.text
            children_ids.append(id_)
            children_names.append(name)

    return (
        Category(id_=category_id, name=category_name, children=children_ids),
        children_ids,
        children_names,
    )


def get_wiki_categories(
    category_id: str = ROOT_CATEGORY_ID,
    category_name: str = ROOT_CATEGORY_NAME,
    depth: int = 1,
):
    queue = deque([(category_id, category_name)])
    seen_ids = {category_id}

    for d in range(depth):
        # Make requests for all items in the queue
        category_ids = []
        category_names = []
        urls = []
        while len(queue) > 0:
            category_id, category_name = queue.popleft()
            category_ids.append(category_id)
            category_names.append(category_name)
            urls.append(
                make_url(
                    base_url=WIKIPEDIA_API_URL,
                    params={
                        "action": "categorytree",
                        "format": "json",
                        "category": f"Category:{category_id}",
                    },
                )
            )
        logging.info("Making %d requests for depth %d", len(urls), d)
        results = asyncio.run(
            make_requests(urls, rate=FLAGS.rate, retries=FLAGS.retries)
        )

        # Process results
        for category_id, category_name, result in zip(
            category_ids, category_names, results
        ):
            category, children_ids, children_names = process_result(
                category_id, category_name, result
            )
            for child_id, child_name in zip(children_ids, children_names):
                if child_id not in seen_ids:
                    queue.append((child_id, child_name))
                    seen_ids.add(child_id)

            yield category


def main(_):
    # Set up
    out_dir = Path(FLAGS.output_dir)
    if out_dir.exists():
        rmtree(out_dir)
    out_dir.mkdir(parents=True)
    logging.get_absl_handler().use_absl_log_file("build", out_dir)

    # Build the tree
    categories = list(get_wiki_categories(depth=FLAGS.depth))

    # Post processing: Remove all cycles
    id_to_category = {category.id_: category for category in categories}

    def find_cycles(node: str, path: tuple[str, ...], cycles: list[tuple[str, ...]]):
        if node in path:
            return cycles + [path[path.index(node) :]]
        else:
            children = id_to_category[node].children if node in id_to_category else []
            for child in children:
                if any(child in cycle for cycle in cycles):
                    continue
                cycles = find_cycles(child, path + (node,), cycles)
            return cycles

    cycles = find_cycles(ROOT_CATEGORY_ID, (), [])

    for cycle in cycles:
        logging.info("Found cycle: %s", " -> ".join(cycle))
        for node, next_node in zip(cycle, cycle[1:] + cycle[:1]):
            id_to_category[node].children.remove(next_node)

    # Post-processing: Remove unreachable nodes
    def find_reachable(node: str, reachable: set[str]):
        reachable.add(node)
        children = id_to_category[node].children if node in id_to_category else []
        for child in children:
            if child not in reachable:
                find_reachable(child, reachable)

    reachable = set()
    find_reachable(ROOT_CATEGORY_ID, reachable)
    unreachable = set(id_to_category.keys()) - reachable
    logging.info("Found %d unreachable nodes: %s", len(unreachable), unreachable)

    categories = list(filter(lambda category: category.id_ in reachable, categories))

    save_categories(categories, FLAGS.output_dir, format="jsonl")
    save_categories(categories, FLAGS.output_dir, format="owl")


if __name__ == "__main__":
    app.run(main)
