"""Prompt templates for constrained VLM inference across visual grounding tasks."""

from typing import Dict, Any

PROMPT_TEMPLATES: Dict[str, str] = {
    "count": (
        "Answer with only one integer.\n"
        "Question: {question}"
    ),
    "yes_no": (
        "Answer with only yes or no.\n"
        "Question: {question}"
    ),
    "attribute": (
        "Answer with only the single attribute word.\n"
        "Question: {question}"
    ),
    "choice": (
        "Question: {question}\n"
        "Options: {options}\n"
        "Answer with only the single letter of the correct option, for example (a) or (b)."
    ),
    "relation": (
        "Question: {question}\n"
        "Answer with only the relation word (e.g., left, right, above, below, front, behind)."
    )
}

def format_prompt(answer_type: str, question: str, options: str = "") -> str:
    """Format prompt with appropriate constraint template."""
    template = PROMPT_TEMPLATES.get(answer_type, PROMPT_TEMPLATES["choice" if options else "count"])
    return template.format(question=question, options=options)
