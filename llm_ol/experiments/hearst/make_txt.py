from pathlib import Path

from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("group_size", 1000, "Number of abstracts per file")
flags.DEFINE_integer("max_depth", None, "Maximum depth of the graph")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "make_txt", flags=FLAGS)

    G = data_model.load_graph(FLAGS.graph_file, FLAGS.max_depth)
    seen = set()
    abstracts = []
    for _, data in G.nodes(data=True):
        for page in data["pages"]:
            if page["id"] in seen:
                continue
            seen.add(page["id"])
            abstracts.append(page["abstract"])

    for i in range(0, len(abstracts), FLAGS.group_size):
        abstract_file = out_dir / f"{i}.txt"
        with open(abstract_file, "w") as f:
            f.write("\n".join(abstracts[i : i + FLAGS.group_size]))

    logging.info("Wrote %d abstracts to %s", len(abstracts), out_dir)


if __name__ == "__main__":
    app.run(main)
