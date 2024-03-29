"""This script is idempotent."""

import asyncio
import json
import random
from pathlib import Path

from absl import app, flags, logging

from llm_ol.experiments.llm.templates import RESPONSE_REGEX
from llm_ol.utils import (
    ParallelAsyncOpenAI,
    load_template,
    setup_logging,
    textpbar,
    wait_for_endpoint,
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "train_dataset", None, "Path to the training dataset", required=True
)
flags.DEFINE_string("test_dataset", None, "Path to the test dataset", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_multi_integer("ports", [], "Ports to use for the API", required=True)
flags.DEFINE_integer(
    "max_concurrent_requests", 512, "Maximum number of concurrent requests per endpoint"
)
flags.DEFINE_integer("k_shot", 5, "Number of samples to provide.")
flags.DEFINE_integer("seed", 42, "Random seed.")


s = """The following is an article's title and abstract. Your task is to assign this article to suitable category hierarchy. \
A category is typically represented by a word or a short phrase, representing broader topics/concepts that the article is about. \
A category hierarchy represented by a collection of paths from the generic root category "Main topic classifications" \
to a specific category suitable for the article.
{% for example in examples %}

### EXAMPLE {{ loop.index }} ###
### ARTICLE ###
Title: {{ example['title'] }}
{{ example['abstract'] }}
### END ARTICLE ###
{% for path in example['paths'] %}
{{ path | join(" -> ") }}
{% endfor %}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}

### ARTICLE ###
Title: {{ title }}
{{ abstract }}
### END ARTICLE ###

Provide a category hierarchy for the above article. Use the same format as the examples above."""
PROMPT_TEMPLATE = load_template(s)


async def query(
    client: ParallelAsyncOpenAI,
    title: str,
    abstract: str,
    examples: list[dict],
    t: float = 0,
) -> str:
    prompt = PROMPT_TEMPLATE.render(title=title, abstract=abstract, examples=examples)
    completion = await client.chat(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
        extra_body={"guided_regex": RESPONSE_REGEX},
        temperature=t,
        max_tokens=2048,
    )
    out = completion.choices[0].message.content
    assert isinstance(out, str)
    return out


async def main(_):
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "categorised_pages.jsonl"

    client = ParallelAsyncOpenAI(
        base_urls=[f"http://localhost:{port}/v1" for port in FLAGS.ports],
        max_concurrent_per_client=FLAGS.max_concurrent_requests,
    )

    with open(FLAGS.train_dataset, "r") as f:
        train_samples = [json.loads(line) for line in f.readlines()]
        train_samples = random.sample(train_samples, FLAGS.k_shot)
    logging.info(
        "Using pages %s as few show examples",
        [sample["id"] for sample in train_samples],
    )

    computed = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))

    with open(FLAGS.test_dataset, "r") as f:
        test_pages = [json.loads(line) for line in f.readlines()]
        test_pages = [
            {
                "id": sample["id"],
                "title": sample["title"],
                "abstract": sample["abstract"],
            }
            for sample in test_pages
            if sample["id"] not in computed
        ]

    pbar = textpbar(len(test_pages))

    async def task(page):
        try:
            out = await query(
                client, page["title"], page["abstract"], examples=train_samples
            )
            with open(out_file, "a") as f:
                f.write(json.dumps({**page, "hierarchy": out}) + "\n")
        except Exception as e:
            logging.error("Error processing page %s: %s", page["id"], repr(e) + str(e))
        finally:
            pbar.update()

    await asyncio.gather(
        *[wait_for_endpoint(f"http://localhost:{port}/health") for port in FLAGS.ports]
    )
    await asyncio.gather(*[task(page) for page in test_pages])


if __name__ == "__main__":
    app.run(lambda _: asyncio.run(main(_)))
