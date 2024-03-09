from openai import AsyncOpenAI

from llm_ol.utils import load_template

#   "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

s = """
<s>[INST] The following is an article's title and abstract. Your task is to assign this article to suitable category hierarchy. \
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

Provide a category hierarchy for the above article. Use the same format as the examples above. [/INST]\
```txt
"""
template = load_template(s)

client = AsyncOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",
)


async def create_hierarchy_v2(title: str, abstract: str, t: float = 0) -> str:
    completion = await client.completions.create(
        model="gpt-3.5-turbo",
        prompt=template.render(title=title, abstract=abstract),
        temperature=t,
        stop=["```"],
        max_tokens=2048,
    )
    return completion.choices[0].text
