"""Prompt template definitions."""

from __future__ import annotations

from jinja2 import Template


SYSTEM_TEMPLATE = """
You are a knowledgeable assistant. Answer the user's question strictly using the provided context.  If the answer
cannot be determined from the context, respond with "I don't know.".
"""

USER_TEMPLATE = """
Context:
{{ context }}

Question: {{ question }}

Answer:
"""


def render_prompt(context: str, question: str) -> str:
    """Render the combined system and user prompt using Jinja2.

    Args:
        context: Concatenated context from retrieved documents.
        question: The user's question.

    Returns:
        The final prompt string to pass to the LLM.
    """
    user_tmpl = Template(USER_TEMPLATE)
    user_content = user_tmpl.render(context=context, question=question)
    return SYSTEM_TEMPLATE.strip() + "\n" + user_content.strip()
