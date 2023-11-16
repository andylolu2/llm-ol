import json
import time
from collections import deque

import requests
from absl import app, flags, logging
from bs4 import BeautifulSoup
from pydantic import BaseModel

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
ROOT_CATEGORY_ID = "Main_topic_classifications"
ROOT_CATEGORY_NAME = "Main topic classifications"

FLAGS = flags.FLAGS
flags.DEFINE_string("output_file", None, "Path to output JSONL file", required=True)
flags.DEFINE_integer("depth", 2, "Depth of categories to fetch")


class Category(BaseModel):
    id_: str
    name: str
    children: list[str] = []


class Requester:
    def __init__(self, sleep_time: float = 0):
        self.session = requests.Session()
        self.sleep_time = sleep_time

    def get(self, url, params):
        time.sleep(self.sleep_time)

        # We won't urlencode the params because the API doesn't like it
        params_str = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"{url}?{params_str}"

        logging.info("GET %s", url)
        return self.session.get(url)


def get_wiki_categories(
    requester: Requester,
    category_id: str = ROOT_CATEGORY_ID,
    category_name: str = ROOT_CATEGORY_NAME,
    depth: int = 1,
):
    def get_category(
        category_id: str, category_name: str
    ) -> tuple[Category, list[str], list[str]]:
        # Sample request
        # https://en.wikipedia.org/w/api.php?action=categorytree&format=json&category=Category:Contents&options={"depth":2}

        response = requester.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "categorytree",
                "format": "json",
                "category": f"Category:{category_id}",
            },
        )
        result = response.json()

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
                id_ = category_tree_link["href"].lstrip("/wiki/Category:")
                name = category_tree_link.text
                children_ids.append(id_)
                children_names.append(name)

        return (
            Category(id_=category_id, name=category_name, children=children_ids),
            children_ids,
            children_names,
        )

    queue = deque([(category_id, category_name)])
    seen_ids = set()

    for _ in range(depth):
        for _ in range(len(queue)):
            category_id, category_name = queue.popleft()
            category, children_ids, children_names = get_category(
                category_id, category_name
            )
            seen_ids.add(category_id)
            for child_id, child_name in zip(children_ids, children_names):
                if child_id not in seen_ids:
                    queue.append((child_id, child_name))

            yield category


def main(_):
    requester = Requester(sleep_time=0.1)

    assert FLAGS.output_file.endswith(".jsonl")

    with open(FLAGS.output_file, "w") as f:
        for category in get_wiki_categories(requester, depth=3):
            f.write(category.model_dump_json() + "\n")


if __name__ == "__main__":
    app.run(main)
