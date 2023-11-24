import graphviz
from absl import app, flags, logging

from llm_ol.dataset.data_model import Category

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "graph_file", None, "Path to input graph file", required=True, short_name="g"
)
flags.DEFINE_string(
    "output_dir", None, "Directory to output graph files", required=True, short_name="o"
)
flags.DEFINE_integer(
    "num_categories", 200, "Number of categories to include", short_name="n"
)
flags.DEFINE_integer(
    "scale", 5, "Scale of graph. Larger means more spread out", short_name="s"
)


def main(_):
    with open(FLAGS.graph_file, "r") as f:
        categories = [Category.model_validate_json(line) for line in f]

    graph = graphviz.Digraph()

    include = set()

    for category in categories[: FLAGS.num_categories]:
        graph.node(category.id_, label=category.name)
        include.add(category.id_)

    for category in categories:
        for child in category.children:
            if category.id_ in include and child in include:
                graph.edge(category.id_, child)

    graph.attr(overlap="false", overlap_scaling=f"-{FLAGS.scale}")
    graph.render(directory=FLAGS.output_dir, format="pdf", engine="sfdp")


if __name__ == "__main__":
    app.run(main)
