"""Text benchmark (Fig. 4): the request is the dataset's prompt, verbatim, as one text part.

The prompt already holds the scenario, the instruction, all four questions with their lettered
options and the JSON output format, so one call answers one item. Nothing is added or removed:
no item ID, no wrapper, no system prompt. A prompt variant (H1) would be a new prompt name with
its own `prompt_version`, so its replies never mix with these.
"""

from __future__ import annotations

from asimov.bench import Benchmark
from asimov.bench.parts import Part, TextPart
from asimov.bench.text_score import score
from asimov.data.text import TextItem, load_text_answers, load_text_items

DATASET_VERSION = "asimov_v2_injuries/0.1.0"
PROMPT_NAME = "text_verbatim"  # the dataset's prompt, unchanged
PROMPTS = (PROMPT_NAME,)


def prompt_version(prompt_name: str) -> str:
    """Everything our code adds beyond the item: here only the dataset version and prompt name."""
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown text prompt {prompt_name!r}; choose from {PROMPTS}")
    return f"{DATASET_VERSION}|{prompt_name}"


def build_request(item: TextItem, prompt_name: str = PROMPT_NAME) -> list[Part]:
    prompt_version(prompt_name)  # rejects unknown names
    return [TextPart(item.prompt)]


BENCHMARK = Benchmark(
    name="text",
    load_items=load_text_items,
    load_answers=load_text_answers,
    build_request=build_request,
    score=score,
    prompt_version=prompt_version,
    columns=(
        ("risk type", "latent_risk_accuracy", True),
        ("severity", "latent_severity_accuracy", True),
        ("effect", "effect_accuracy", True),
        ("activated", "activated_risk_accuracy", True),
        ("parse fail", "parse_failure_rate", True),
    ),
)
