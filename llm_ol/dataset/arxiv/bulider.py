from absl import app, flags
from arxiv.taxonomy.definitions import ARCHIVES_ACTIVE as ARCHIVES
from arxiv.taxonomy.definitions import CATEGORIES_ACTIVE as CATEGORIES
from arxiv.taxonomy.definitions import CATEGORY_ALIASES, GROUPS

from llm_ol.dataset.data_model import Category

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_file", None, "Path to output JSONL file", required=True, short_name="o"
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
    pruned = set()
    for node_id, children in node_to_children.items():
        new_children = children.copy()
        for child_id in children:
            if node_names[node_id] == node_names[child_id]:
                new_children |= node_to_children[child_id]
                pruned.add(child_id)
        node_to_children[node_id] = new_children
    nodes = set(node_to_children.keys()) - pruned

    with open(FLAGS.output_file, "w") as f:
        for node_id in nodes:
            name = node_names[node_id]
            children = list(node_to_children[node_id])
            category = Category(id_=node_id, name=name, children=children)
            f.write(category.model_dump_json() + "\n")


if __name__ == "__main__":
    app.run(main)
