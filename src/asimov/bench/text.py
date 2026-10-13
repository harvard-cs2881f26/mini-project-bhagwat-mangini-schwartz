"""Text benchmark (Fig. 4): the request is the dataset's prompt, verbatim, as one text part.

The prompt already holds the scenario, the instruction, all four questions with their lettered
options and the JSON output format, so one call answers one item. Nothing is added or removed:
no item ID, no wrapper, no system prompt. A prompt variant (H1) would be a new prompt name with
its own `prompt_version`, so its replies never mix with these.

Decision prompts (Part 2) send one DecisionPart instead: the item's scenario and instruction as
the state, and its four questions with the paper's options, worded from
prompts/decision_questions.yaml (the one home for that wording).
"""

from __future__ import annotations

import hashlib
import re
from functools import cache
from pathlib import Path

import yaml

from asimov.bench import Benchmark
from asimov.bench.parts import DecisionPart, DecisionQuestion, Part, TextPart
from asimov.bench.text_score import score
from asimov.data.text import ParsedPrompt, Question, TextItem, load_text_answers, load_text_items, parse_prompt

DATASET_VERSION = "asimov_v2_injuries/0.1.0"
PROMPT_NAME = "text_verbatim"  # the dataset's prompt, unchanged
# decision_<wording>[_letters]: wording = the paper's full text per question or the short text in the
# wording file; option values are short names (e.g. "severe") unless `_letters` (A-E). WORKLOG 2026-10-12.
DECISION_PROMPTS = ("decision_short", "decision_full", "decision_short_letters")
PROMPTS = (PROMPT_NAME, *DECISION_PROMPTS)
WORDING = Path(__file__).resolve().parents[1] / "prompts" / "decision_questions.yaml"


def prompt_version(prompt_name: str) -> str:
    """Everything our code adds beyond the item: the dataset version and prompt name, plus for
    decision prompts the hash of the wording file."""
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown text prompt {prompt_name!r}; choose from {PROMPTS}")
    if prompt_name in DECISION_PROMPTS:
        return f"{DATASET_VERSION}|{prompt_name}|{hashlib.sha256(WORDING.read_bytes()).hexdigest()[:12]}"
    return f"{DATASET_VERSION}|{prompt_name}"


@cache
def wording() -> dict:
    return yaml.safe_load(WORDING.read_text())


def option_name(text: str) -> str:
    """Q1 option text -> a short value name: 'Burns, Chemical' -> 'burns_chemical'."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def option_values(question: Question, prompt_name: str) -> list[str]:
    """Values sent for each option, in prompt order: letters, Q1 names made from the option text, or
    the fixed names for Q2-Q4 from the wording file."""
    if prompt_name.endswith("_letters"):
        return list(question.options)
    if question.key in wording()["names"]:
        return [wording()["names"][question.key][letter] for letter in question.options]
    return [option_name(text) for text in question.options.values()]


def decision_request(parsed: ParsedPrompt, prompt_name: str) -> DecisionPart:
    """One DecisionPart: the state, and each question's instructions and the paper's options."""
    w = wording()
    state = w["state"].format(scenario=parsed.scenario, instruction=parsed.instruction)
    questions = []
    for q in parsed.questions:
        values = option_values(q, prompt_name)
        if len(set(values)) != len(values):
            raise ValueError(f"{q.key}: option values aren't unique: {values}")
        questions.append(DecisionQuestion(
            id=q.key, instructions=q.text if prompt_name.startswith("decision_full") else w["short"][q.key],
            options=tuple(zip(values, q.options.values(), strict=True)), letters=tuple(q.options)))
    return DecisionPart(state=state, questions=tuple(questions))


def build_request(item: TextItem, prompt_name: str = PROMPT_NAME) -> list[Part]:
    prompt_version(prompt_name)  # rejects unknown names
    if prompt_name in DECISION_PROMPTS:
        return [decision_request(parse_prompt(item.prompt), prompt_name)]
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
