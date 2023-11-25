from pathlib import Path
from shutil import rmtree

from absl import app, flags, logging
from arxiv.taxonomy.definitions import ARCHIVES_ACTIVE as ARCHIVES
from arxiv.taxonomy.definitions import CATEGORIES_ACTIVE as CATEGORIES
from arxiv.taxonomy.definitions import CATEGORY_ALIASES, GROUPS

from llm_ol.dataset.data_model import Category, save_categories

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)


def group_id(id_: str) -> str:
    return f"group-{id_}"


def archive_id(id_: str) -> str:
    return f"archive-{id_}"


def category_id(id_: str) -> str:
    return f"category-{id_}"


def main(_):
    """
    The arXiv taxonomy is a hierarchical classification of arXiv papers into three
    levels: groups -> archives -> categories.

    The taxonomy is hard-coded in the arxiv.taxonomy.definitions module.
    """

    # Set up
    out_dir = Path(FLAGS.output_dir)
    if out_dir.exists():
        rmtree(out_dir)
    out_dir.mkdir(parents=True)
    logging.get_absl_handler().use_absl_log_file("build", out_dir)

    # Build the tree
    node_names = {"root": "Root"}  # node_id -> name
    node_to_children = {"root": set()}  # node_id -> set(child_id)

    for key, value in GROUPS.items():
        if value.get("is_test", False):
            continue
        node_to_children["root"].add(group_id(key))
        node_names[group_id(key)] = value["name"]
        node_to_children[group_id(key)] = set()

    for key, value in ARCHIVES.items():
        if value.get("is_test", False):
            continue
        node_to_children[group_id(value["in_group"])].add(archive_id(key))
        node_names[archive_id(key)] = value["name"]
        node_to_children[archive_id(key)] = set()

    for key, value in CATEGORIES.items():
        if key in CATEGORY_ALIASES:
            key = CATEGORY_ALIASES[key]
        node_to_children[archive_id(value["in_archive"])].add(category_id(key))
        node_names[category_id(key)] = value["name"]
        node_to_children[category_id(key)] = set()

    # Post-processing: Shorten all paths with repeated entries
    # E.g. A -> B -> B -> C becomes A -> B -> C
    def contract(root: str, seen: set[str]) -> None:
        children = node_to_children[root].copy()
        for child in children:
            if child not in seen:
                seen.add(child)
                contract(child, seen)

            if node_names[root] == node_names[child]:
                logging.info(
                    "Contracting %s -> %s (Name: %s)", child, root, node_names[root]
                )
                node_to_children[root] |= node_to_children[child]
                node_to_children[root].remove(child)

    contract("root", set())

    # Post-processing: Remove unreachable nodes
    def find_reachable(node: str, reachable: set[str]):
        reachable.add(node)
        for child in node_to_children[node]:
            if child not in reachable:
                find_reachable(child, reachable)

    reachable = set()
    find_reachable("root", reachable)

    unreachable = set(node_to_children.keys()) - reachable
    logging.info("Found %d unreachable nodes: %s", len(unreachable), unreachable)

    categories = [
        Category(id_=node, name=node_names[node], children=list(children))
        for node, children in node_to_children.items()
        if node in reachable
    ]

    save_categories(categories, out_dir, "jsonl")
    save_categories(categories, out_dir, "owl")


if __name__ == "__main__":
    app.run(main)
