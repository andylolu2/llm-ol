import re

from llm_ol.utils import load_template

_PROMPT_TEMPLATE = """Title: {{ title }}
{{ abstract }}"""

_RESPONSE_TEMPLATE = """{% for path in paths %}
{{ path | join(" -> ") }}
{% endfor %}"""

MISTRAL_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
PROMPT_TEMPLATE = load_template(_PROMPT_TEMPLATE)
RESPONSE_TEMPLATE = load_template(_RESPONSE_TEMPLATE)
RESPONSE_REGEX = r"(Main topic classifications( -> [\w()\-\–\—,.?!/\\&\"\'+=\[\]\{\} ]+)+\n)*(Main topic classifications( -> [\w()\-\–\—,.?!/\\&\"\'+=\[\]\{\} ]+)+)\n?"
