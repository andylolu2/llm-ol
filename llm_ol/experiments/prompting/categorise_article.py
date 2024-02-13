import guidance
from guidance import gen, instruction

from llm_ol.utils import load_template

s = """
The following is an article's title and abstract. Your task is to assign this article to suitable category(ies). \
A category is typically represented by a word or a short phrase, representing broader topics/concepts that the article is about.

Title: {{ title }}
{{ abstract }}

{% if categories %}
Below is a list of {{ categories | length }} possible category(ies) that you can choose from. \
Alternatively, you can also create new category(ies) if you think it is necessary:
{% for category in categories %}
{{ loop.index }}. {{ category }}
{% endfor %}
{% endif %}
"""


@guidance
def categorise_article(
    lm: guidance.models.Model,
    title: str,
    abstract: str,
    categories: list[str],
    t: float = 0,
) -> guidance.models.Model:
    with instruction():
        lm += load_template(s).render(
            title=title, abstract=abstract, categories=categories
        )
    lm += (
        "Here are "
        + gen(name="n", regex=r"\d+", temperature=t)
        + " category(ies) that are suitable for this article:\n"
    )
    for i in range(int(lm["n"])):
        lm += (
            f"{i+1}. "
            + gen(name="cats", list_append=True, stop=["\n", ".", ":"], temperature=t)
            + "\n"
        )
    return lm
