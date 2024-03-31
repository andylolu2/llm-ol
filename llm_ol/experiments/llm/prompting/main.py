"""This script is idempotent."""

import json
import random
from pathlib import Path

from absl import app, flags, logging
from vllm import LLM, SamplingParams

from llm_ol.utils import batch, load_template, setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "train_dataset", None, "Path to the training dataset", required=True
)
flags.DEFINE_string("test_dataset", None, "Path to the test dataset", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("k_shot", 1, "Number of samples to provide.")
flags.DEFINE_integer("seed", 0, "Random seed.")


s = """The following is an article's title and abstract. Your task is to assign this article to suitable category hierarchy. \
A category is typically represented by a word or a short phrase, representing broader topics/concepts that the article is about. \
A category hierarchy represented by a collection of paths from the generic root category "Main topic classifications" \
to a specific category suitable for the article. The topics titles should become more and more specific as you move from the root to the leaf. \
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


def main(_):
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "categorised_pages.jsonl"

    with open(FLAGS.train_dataset, "r") as f:
        examples = [json.loads(line) for line in f.readlines()]
    logging.info(
        "Example prompt:\n%s",
        PROMPT_TEMPLATE.render(
            title="TITLE",
            abstract="ABSTRACT",
            examples=random.sample(examples, FLAGS.k_shot),
        ),
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
    logging.info("Computing responses for %d pages", len(test_pages))

    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        download_dir="out/models",
        max_num_seqs=512,
        max_paddings=512,
    )
    tokenizer = llm.get_tokenizer()

    for pages in batch(test_pages, 1000):
        prompts = []
        few_shot_examples = [random.sample(examples, FLAGS.k_shot) for _ in pages]
        for page, few_shot in zip(pages, few_shot_examples):
            messages = [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.render(
                        title=page["title"],
                        abstract=page["abstract"],
                        examples=few_shot,
                    ),
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=1024,
                repetition_penalty=1.15,
                stop=["\n\n"],
            ),
        )
        for page, few_shot, out in zip(pages, few_shot_examples, outputs):
            with open(out_file, "a") as f:
                response = out.outputs[0].text
                item = {
                    **page,
                    "hierarchy": response,
                    "few_shot_ids": [ex["id"] for ex in few_shot],
                }
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    app.run(main)
