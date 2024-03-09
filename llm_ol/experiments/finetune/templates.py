from llm_ol.utils import load_template

_PROMPT_TEMPLATE = """
Title: {{ title }}
{{ abstract }}"""

_RESPONSE_TEMPLATE = """
{% for path in paths %}
{{ path | join(" -> ") }}
{% endfor %}"""

PROMPT_TEMPLATE = load_template(_PROMPT_TEMPLATE)
RESPONSE_TEMPLATE = load_template(_RESPONSE_TEMPLATE)
