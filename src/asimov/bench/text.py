"""Text benchmark (Fig. 4): the request is the dataset's prompt, verbatim, as one text part.

The prompt already holds the scenario, the instruction, all four questions with their lettered
options and the JSON output format, so one call answers one item. Nothing is added or removed:
no item ID, no wrapper, no system prompt.

Prompt variants (not for the reproduction) add text after the prompt, kept in
`experiments/<name>/suffix.txt`; the added text is part of `prompt_version`, so their replies
never mix with the verbatim prompt's.
"""

from __future__ import annotations

from pathlib import Path

from asimov.bench import Benchmark
from asimov.bench.parts import Part, TextPart
from asimov.bench.text_score import score
from asimov.data.text import TextItem, load_text_answers, load_text_items, load_text_options

DATASET_VERSION = "asimov_v2_injuries/0.1.0"
PROMPT_NAME = "text_verbatim"  # the dataset's prompt, unchanged (the reproduction)
LETTER_ONLY = "text_letter_only"  # + "write only the option letter" (experiments/text_letter_only)
PROMPTS = (PROMPT_NAME, LETTER_ONLY)
EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"


def suffix(prompt_name: str) -> str:
    """Text added after the dataset's prompt: none for the verbatim prompt."""
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown text prompt {prompt_name!r}; choose from {PROMPTS}")
    return "" if prompt_name == PROMPT_NAME else (EXPERIMENTS / prompt_name / "suffix.txt").read_text().strip()


def prompt_version(prompt_name: str) -> str:
    """Everything our code adds beyond the item: the dataset version, prompt name and any added text."""
    added = suffix(prompt_name)
    return f"{DATASET_VERSION}|{prompt_name}" + (f"|{added}" if added else "")


def build_request(item: TextItem, prompt_name: str = PROMPT_NAME) -> list[Part]:
    added = suffix(prompt_name)  # rejects unknown names
    return [TextPart(item.prompt + (f"\n\n{added}" if added else ""))]


def score_text(answers: dict, replies: dict[str, str]) -> dict:
    """The scorer, given each item's option text so answers written as text are read as letters."""
    return score(answers, replies, options=load_text_options())


BENCHMARK = Benchmark(
    name="text",
    load_items=load_text_items,
    load_answers=load_text_answers,
    build_request=build_request,
    score=score_text,
    prompt_version=prompt_version,
    columns=(
        ("risk type", "latent_risk_accuracy", True),
        ("severity", "latent_severity_accuracy", True),
        ("effect", "effect_accuracy", True),
        ("activated", "activated_risk_accuracy", True),
        ("parse fail", "parse_failure_rate", True),
    ),
)
