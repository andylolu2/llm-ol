import guidance
from guidance import gen, instruction

from llm_ol.utils import load_template

s = """
The following is an article's title and abstract. Your task is to assign this article to suitable category hierarchy. \
A category is typically represented by a word or a short phrase, representing broader topics/concepts that the article is about. \
A category hierarchy is a directed acyclic graph that starts with a detailed categorisation and becomes more and more \
general higher up the hierarchy, until it reaches the special base category "ROOT".

An example hierarchy for an article on "Addition" might be have the following category hierarchy:

```json
{
    "ROOT": {
        "Mathematics": {
            "Mathematical notation": "LEAF"
        },
        "Entities": {
            "Systems": {
                "Notation": {
                    "Mathematical notation": "LEAF"
                }
            }
        }
    }
}

Title: {{ title }}
{{ abstract }}
"""


@guidance
def create_hierarchy(
    lm: guidance.models.Model,
    title: str,
    abstract: str,
    t: float = 0,
) -> guidance.models.Model:
    with instruction():
        lm += load_template(s).render(title=title, abstract=abstract)
    lm += "```json\n" + gen(
        name="hierarchy", max_tokens=1000, stop="```", temperature=t
    )
    return lm
