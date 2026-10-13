"""Text benchmark (Fig. 4): the request is the dataset's prompt, verbatim, as one text part.

The prompt already holds the scenario, the instruction, all four questions with their lettered
options and the JSON output format, so one call answers one item. Nothing is added or removed:
no item ID, no wrapper, no system prompt. A prompt variant (H1) would be a new prompt name with
its own `prompt_version`, so its replies never mix with these.

Decision prompts (Part 2) send one DecisionPart instead: the item's scenario and instruction as
the state, and its four questions with the paper's options, worded from
prompts/decision_questions.yaml (the one home for that wording). Their names are
`decision_<wording>[_letters][_<mode>]` (parse_decision_prompt): wording `short` or `full`,
`_letters` for A-E option values, and `_cumulative` (none = independent, non-cumulative) for
whether a question sees the model's own earlier picks (models.decision.generate_decision).
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import cache
from pathlib import Path
from typing import NamedTuple

import yaml

from asimov.bench import Benchmark
from asimov.bench.parts import MODES, DecisionPart, DecisionQuestion, Mode, Part, TextPart
from asimov.bench.text_score import score
from asimov.data.text import ParsedPrompt, Question, TextItem, load_text_answers, load_text_items, parse_prompt

DATASET_VERSION = "asimov_v2_injuries/0.1.0"
PROMPT_NAME = "text_verbatim"  # the dataset's prompt, unchanged
# decision_<wording>[_letters][_<mode>]: wording = the paper's full text per question or the short text
# in the wording file; option values are short names (e.g. "severe") unless `_letters` (A-E); mode
# `_cumulative` passes the model's own earlier picks to later questions (none = independent, non-cumulative).
# WORKLOG 2026-10-12 ("Decision-model wording", "Decision question modes").
WORDINGS = ("short", "full")
DECISION_NAME = re.compile(r"decision_(short|full)(_letters)?(?:_(cumulative))?")


class DecisionPrompt(NamedTuple):
    wording: str  # "short" | "full"
    letters: bool  # option values are the letters A-E
    mode: Mode


def decision_prompt_name(words: str, letters: bool, mode: Mode) -> str:
    return "decision_" + words + ("_letters" if letters else "") + ("" if mode == "independent" else f"_{mode}")


def parse_decision_prompt(name: str) -> DecisionPrompt:
    """'decision_short_letters_cumulative' -> ('short', True, 'cumulative'); ValueError if not a decision prompt."""
    m = DECISION_NAME.fullmatch(name)
    if not m:
        raise ValueError(f"not a decision prompt: {name!r}")
    return DecisionPrompt(m[1], bool(m[2]), m[3] or "independent")


DECISION_PROMPTS = tuple(decision_prompt_name(w, letters, mode) for mode in MODES for w in WORDINGS for letters in (False, True))
PROMPTS = (PROMPT_NAME, *DECISION_PROMPTS)
WORDING = Path(__file__).resolve().parents[1] / "prompts" / "decision_questions.yaml"
# Sections of the wording file a decision request uses; their content (not the file's bytes, so
# comments don't count) is hashed into prompt_version. `context` only matters to cumulative.
REQUEST_SECTIONS = ("state", "short", "names")


def prompt_version(prompt_name: str) -> str:
    """Everything our code adds beyond the item: the dataset version and prompt name, plus for
    decision prompts a hash of the wording sections the request uses."""
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown text prompt {prompt_name!r}; choose from {PROMPTS}")
    if prompt_name in DECISION_PROMPTS:
        sections = REQUEST_SECTIONS
        if parse_decision_prompt(prompt_name).mode != "independent":
            sections = (*sections, "context")
        used = json.dumps({k: wording()[k] for k in sections}, sort_keys=True)
        return f"{DATASET_VERSION}|{prompt_name}|{hashlib.sha256(used.encode()).hexdigest()[:12]}"
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
    if parse_decision_prompt(prompt_name).letters:
        return list(question.options)
    if question.key in wording()["names"]:
        return [wording()["names"][question.key][letter] for letter in question.options]
    return [option_name(text) for text in question.options.values()]


def decision_request(parsed: ParsedPrompt, prompt_name: str) -> DecisionPart:
    """One DecisionPart: the state, each question's instructions and the paper's options, and the
    question mode with each question's context line (used only by cumulative)."""
    w, prompt = wording(), parse_decision_prompt(prompt_name)
    state = w["state"].format(scenario=parsed.scenario, instruction=parsed.instruction)
    questions = []
    for q in parsed.questions:
        values = option_values(q, prompt_name)
        if len(set(values)) != len(values):
            raise ValueError(f"{q.key}: option values aren't unique: {values}")
        questions.append(DecisionQuestion(
            id=q.key, instructions=q.text if prompt.wording == "full" else w["short"][q.key],
            options=tuple(zip(values, q.options.values(), strict=True)), letters=tuple(q.options),
            context=w["context"].get(q.key, "")))
    return DecisionPart(state=state, questions=tuple(questions), mode=prompt.mode)


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
