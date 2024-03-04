from openai import AsyncOpenAI

from llm_ol.utils import load_template

s = """
The following is an article's title and abstract. Your task is to assign this article to suitable category hierarchy. \
A category is typically represented by a word or a short phrase, representing broader topics/concepts that the article is about. \
A category hierarchy is a directed acyclic graph that starts with a detailed categorisation and becomes more and more \
general higher up the hierarchy, until it reaches the special base category "ROOT".

An example hierarchy for an article on "Single whip law" might be have the following category hierarchy:
```txt
Main topic classifications -> Economy -> Economic history -> History of taxation
Main topic classifications -> Law -> Law by issue -> Legal history by issue -> History of taxation
Main topic classifications -> Law -> Law by issue -> Tax law
Main topic classifications -> Law -> Law stubs -> Asian law stubs
Main topic classifications -> Politics -> Political history -> History of taxation
```

Another example hierarchy for an article on "Stoning" is:
```txt
Main topic classifications -> Human behavior -> Abuse -> Cruelty -> Torture
Main topic classifications -> Human behavior -> Violence -> Torture
Main topic classifications -> Law -> Law-related events -> Crimes -> Torture
Main topic classifications -> Law -> Legal aspects of death -> Killings by type
Main topic classifications -> Society -> Violence -> Torture
```

### ARTICLE ###
Title: {{ title }}
{{ abstract }}
### END ARTICLE ###

Provide a category hierarchy for the above article. Use the same format as the examples above.
"""

client = AsyncOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",
)


async def create_hierarchy_v2(title: str, abstract: str, t: float = 0) -> str:
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": load_template(s).render(title=title, abstract=abstract),
            },
        ],
        temperature=t,
        max_tokens=1024,
    )
    return completion.choices[0].message.content
