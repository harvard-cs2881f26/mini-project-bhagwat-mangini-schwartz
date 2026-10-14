"""Text benchmark (Fig. 4): the request is the dataset's prompt, verbatim, as one text part.

The prompt already holds the scenario, the instruction, all four questions with their lettered
options and the JSON output format, so one call answers one item. Nothing is added or removed:
no item ID, no wrapper, no system prompt.

Prompt variants (not for the reproduction) add text after the prompt, kept in
`experiments/<name>/suffix.txt`, and/or before it, kept in `experiments/<name>/prefix.txt` (each
optional, joined to the prompt by a blank line); the added text is part of `prompt_version`, so
their replies never mix with the verbatim prompt's.

Decision prompts (Part 2) send one DecisionPart instead: the item's scenario and instruction as
the state, and its four questions with the paper's options, worded from
prompts/decision_questions.yaml (the one home for that wording). Their names are
`decision_<wording>[_letters][_<mode>]` (parse_decision_prompt): wording `short` or `full`,
`_letters` for A-E option values, and `_cumulative` (none = independent, non-cumulative) for
whether a question sees the model's own earlier picks (models.decision.generate_decision).
A trailing `_local` (e.g. decision_short_local) marks the same request for a local model
(models.local_hf, models.nli), which renders it as text with the wording file's `local` section;
that section is hashed into the `_local` versions only, so the hosted versions never move.
`_oneshot_local` (e.g. decision_short_oneshot_local) adds one made-up, answered example per question
before the real one, from the wording file's `oneshot` section (hashed into those versions only).
Hint variants (Part 2 prompt experiments, EXPLORATORY; WORKLOG 2026-10-14) are text prompts named by a folder
`experiments/<name>/` holding `variant.yaml`: a saved run and model whose picks go into each item's request,
read from that run's replies under its summary's key with the text scorer's own reading (`reply_letters`), and a
template. Kind `hints` sends the verbatim prompt, a blank line, then the template filled with the picks as
"<letter>. <the paper's option text>"; kind `q2_q4` sends the verbatim prompt without Q1 and Q1's JSON keys, the
template (filled with the Q1 pick's option text) taking Q1's place. Their version hashes the variant file, the
template and the source run's key.

A name is a decision prompt only if parse_decision_prompt accepts it; a hint variant if its folder has a
variant.yaml; every other name is a text prompt (verbatim or a suffix variant).
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import cache
from pathlib import Path
from typing import Literal, NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict

from asimov.bench import Benchmark
from asimov.bench.parts import MODES, DecisionExample, DecisionPart, DecisionQuestion, Mode, Part, TextPart
from asimov.bench.text_score import QUESTIONS, reply_letters, score
from asimov.data.text import (
    ParsedPrompt,
    Question,
    TextItem,
    load_text_answers,
    load_text_items,
    load_text_options,
    parse_options,
    parse_prompt,
)
from asimov.run_ids import folder_name, read_ids

DATASET_VERSION = "asimov_v2_injuries/0.1.0"
PROMPT_NAME = "text_verbatim"  # the dataset's prompt, unchanged (the reproduction)
LETTER_ONLY = "text_letter_only"  # + "write only the option letter" (experiments/text_letter_only)
FEWSHOT_FORMAT = "text_fewshot_format"  # worked examples before, format rules after (EXPLORATORY)
TEXT_PROMPTS = (PROMPT_NAME, LETTER_ONLY, FEWSHOT_FORMAT)
EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"
RESULTS = EXPERIMENTS.parent / "results"
VARIANT_FILE = "variant.yaml"  # marks experiments/<name>/ as a hint variant
HINT_PROMPTS = tuple(sorted(p.parent.name for p in EXPERIMENTS.glob(f"*/{VARIANT_FILE}")))
NO_HINT = "no answer"  # shown for a question the source model left unanswered
# Q1's keys in the dataset's JSON format line (the same on all 319 prompts), removed by kind q2_q4
Q1_JSON_KEYS = "'latent_risk_response': latent_risk_response, 'latent_risk_rationale': latent_risk_rationale,\n "
Q2_START = re.compile(r"^Q2 \(", re.M)
# decision_<wording>[_letters][_<mode>]: wording = the paper's full text per question or the short text
# in the wording file; option values are short names (e.g. "severe") unless `_letters` (A-E); mode
# `_cumulative` passes the model's own earlier picks to later questions (none = independent, non-cumulative);
# a trailing `_local` = rendered as text for a local model (WORKLOG 2026-10-13, "Local decision models").
# WORKLOG 2026-10-12 ("Decision-model wording", "Decision question modes").
WORDINGS = ("short", "full")
LOCAL = "_local"
ONESHOT = "_oneshot_local"  # one-shot is for local base models only
DECISION_NAME = re.compile(r"decision_(short|full)(_letters)?(?:_(cumulative))?(?:(_oneshot)?(_local))?")


class DecisionPrompt(NamedTuple):
    wording: str  # "short" | "full"
    letters: bool  # option values are the letters A-E
    mode: Mode


def decision_prompt_name(words: str, letters: bool, mode: Mode) -> str:
    return "decision_" + words + ("_letters" if letters else "") + ("" if mode == "independent" else f"_{mode}")


def parse_decision_prompt(name: str) -> DecisionPrompt:
    """'decision_short_letters_cumulative' -> ('short', True, 'cumulative'); ValueError if not a decision prompt.
    A `_local` name parses like the name without it (is_local_prompt tells them apart)."""
    m = DECISION_NAME.fullmatch(name)
    if not m:
        raise ValueError(f"not a decision prompt: {name!r}")
    return DecisionPrompt(m[1], bool(m[2]), m[3] or "independent")


DECISION_PROMPTS = tuple(decision_prompt_name(w, letters, mode) for mode in MODES for w in WORDINGS for letters in (False, True))
LOCAL_DECISION_PROMPTS = tuple(p + LOCAL for p in DECISION_PROMPTS)
ONESHOT_PROMPTS = tuple(p + ONESHOT for p in DECISION_PROMPTS)
PROMPTS = (*TEXT_PROMPTS, *HINT_PROMPTS, *DECISION_PROMPTS, *LOCAL_DECISION_PROMPTS, *ONESHOT_PROMPTS)
WORDING = Path(__file__).resolve().parents[1] / "prompts" / "decision_questions.yaml"
# Sections of the wording file a decision request uses; their content (not the file's bytes, so
# comments don't count) is hashed into prompt_version. `context` only matters to cumulative, `local`
# only to `_local` prompts.
REQUEST_SECTIONS = ("state", "short", "names")


def added_text(prompt_name: str, where: str) -> str:
    """experiments/<prompt>/<where>.txt, stripped, or "" if the variant has none (or is the verbatim prompt)."""
    if prompt_name not in TEXT_PROMPTS:
        raise ValueError(f"unknown text prompt {prompt_name!r}; choose from {PROMPTS}")
    path = EXPERIMENTS / prompt_name / f"{where}.txt"
    return "" if prompt_name == PROMPT_NAME or not path.exists() else path.read_text().strip()


def suffix(prompt_name: str) -> str:
    """Text added after the dataset's prompt: none for the verbatim prompt."""
    return added_text(prompt_name, "suffix")


def prefix(prompt_name: str) -> str:
    """Text added before the dataset's prompt: none for the verbatim prompt."""
    return added_text(prompt_name, "prefix")


def is_decision_prompt(prompt_name: str) -> bool:
    """True for exactly the names parse_decision_prompt accepts (DECISION_, LOCAL_DECISION_ or ONESHOT_PROMPTS)."""
    return DECISION_NAME.fullmatch(prompt_name) is not None


def is_local_prompt(prompt_name: str) -> bool:
    """A decision prompt for a local model: rendered as text with the wording file's `local` section."""
    m = DECISION_NAME.fullmatch(prompt_name)
    return bool(m and m[5])


def is_oneshot_prompt(prompt_name: str) -> bool:
    """A local decision prompt with one made-up, answered example per question (wording file `oneshot`)."""
    m = DECISION_NAME.fullmatch(prompt_name)
    return bool(m and m[4])


class HintSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str  # the saved run's name (config `run`), never a folder name
    model: str  # the model id in that run's config


class VariantFile(BaseModel):
    """experiments/<name>/variant.yaml (unknown keys fail)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["hints", "q2_q4"]
    source: HintSource
    template: str  # file name in the variant's folder


class Variant(NamedTuple):
    name: str
    kind: str
    source: HintSource
    template: str  # the template's text


@cache
def variant(prompt_name: str) -> Variant | None:
    """The hint variant named prompt_name, or None if experiments/<name>/variant.yaml doesn't exist."""
    path = EXPERIMENTS / prompt_name / VARIANT_FILE
    if not path.is_file():
        return None
    spec = VariantFile.model_validate(yaml.safe_load(path.read_text()))
    return Variant(prompt_name, spec.kind, spec.source, (path.parent / spec.template).read_text().strip())


def is_hint_variant(prompt_name: str) -> bool:
    return variant(prompt_name) is not None


@cache
def saved_replies(source: HintSource) -> tuple[str, dict[str, str]]:
    """(cache key, item id -> reply text) of a saved run's model: only replies under its summary's key."""
    ids = read_ids(RESULTS)
    if source.run not in ids:
        raise ValueError(f"hint source run {source.run!r} has no folder under results/ (results/run_ids.yaml)")
    mdir = RESULTS / folder_name(ids[source.run], source.run) / source.model
    key = json.loads((mdir / "summary.json").read_text())["key"]
    rows = [json.loads(line) for line in (mdir / "responses.jsonl").read_text().splitlines() if line.strip()]
    return key, {r["id"]: r["text"] for r in rows if r["key"] == key}


def item_hints(item: TextItem, source: HintSource) -> dict[str, str | None]:
    """The source model's letter per question for this item (None = unanswered), read as the scorer reads it."""
    _, replies = saved_replies(source)
    if item.id not in replies:
        raise ValueError(f"{source.run}/{source.model} has no saved reply for {item.id}; rerun that config first")
    return reply_letters(replies[item.id], parse_options(item.prompt))


def without_q1(prompt: str, statement: str) -> str:
    """The verbatim prompt with the Q1 block replaced by `statement` and Q1's keys cut from the JSON format.
    Q2-Q4, the scenario, the instruction and the rest of the format stay verbatim."""
    parse_prompt(prompt)  # fails loudly if the prompt isn't the dataset's template
    q2 = list(Q2_START.finditer(prompt))
    if prompt.count("Q1.") != 1 or len(q2) != 1 or prompt.count(Q1_JSON_KEYS) != 1:
        raise ValueError("prompt doesn't have exactly one Q1 block, one Q2 heading and one Q1 JSON entry")
    start, end = prompt.index("Q1."), q2[0].start()
    return (prompt[:start] + statement + "\n\n" + prompt[end:]).replace(Q1_JSON_KEYS, "")


def hint_request(item: TextItem, v: Variant) -> str:
    letters, options = item_hints(item, v.source), parse_options(item.prompt)
    if v.kind == "hints":
        shown = {q: NO_HINT if (x := letters[q]) is None else f"{x}. {options[q][x]}" for q in QUESTIONS}
        return item.prompt + "\n\n" + v.template.format(**shown)
    q1 = letters[QUESTIONS[0]]
    injury = NO_HINT if q1 is None else options[QUESTIONS[0]][q1]
    return without_q1(item.prompt, v.template.format(**{QUESTIONS[0]: injury}))


def variant_version(v: Variant) -> str:
    key, _ = saved_replies(v.source)
    spec = {"kind": v.kind, "source": v.source.model_dump(), "template": v.template, "source_key": key}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    return f"{DATASET_VERSION}|{v.name}|{digest}"


def prompt_version(prompt_name: str) -> str:
    """Everything our code adds beyond the item: the dataset version and prompt name, plus any added
    text (prompt variants), a hash of a hint variant's spec, template and source key, or, for decision
    prompts, a hash of the wording sections the request uses."""
    if (v := variant(prompt_name)) is not None:
        return variant_version(v)
    if is_decision_prompt(prompt_name):
        sections = REQUEST_SECTIONS
        if parse_decision_prompt(prompt_name).mode != "independent":
            sections = (*sections, "context")
        if is_local_prompt(prompt_name):
            sections = (*sections, "local")
        if is_oneshot_prompt(prompt_name):
            sections = (*sections, "oneshot")
        used = json.dumps({k: wording()[k] for k in sections}, sort_keys=True)
        return f"{DATASET_VERSION}|{prompt_name}|{hashlib.sha256(used.encode()).hexdigest()[:12]}"
    added, before = suffix(prompt_name), prefix(prompt_name)  # rejects unknown names
    return f"{DATASET_VERSION}|{prompt_name}" + (f"|{added}" if added else "") + (f"|prefix:{before}" if before else "")


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
        questions.append(
            DecisionQuestion(
                id=q.key,
                instructions=q.text if prompt.wording == "full" else w["short"][q.key],
                options=tuple(zip(values, q.options.values(), strict=True)),
                letters=tuple(q.options),
                context=w["context"].get(q.key, ""),
                example=example(q.key, tuple(q.options.values())) if is_oneshot_prompt(prompt_name) else None,
            )
        )
    return DecisionPart(state=state, questions=tuple(questions), mode=prompt.mode)


def example(question: str, options: tuple[str, ...]) -> DecisionExample:
    """The made-up worked example for a question: its own scenario, the item's own option texts
    (Q1, whose options differ per item, gets the example's own), and the example's answer letter."""
    w = wording()
    shot = w["oneshot"]
    texts = tuple(shot["latent_risk_options"]) if question == "latent_risk" else options
    return DecisionExample(state=w["state"].format(**shot["example"]), options=texts, answer=shot["answers"][question])


def build_request(item: TextItem, prompt_name: str = PROMPT_NAME) -> list[Part]:
    if is_decision_prompt(prompt_name):
        return [decision_request(parse_prompt(item.prompt), prompt_name)]
    if (v := variant(prompt_name)) is not None:
        return [TextPart(hint_request(item, v))]
    added, before = suffix(prompt_name), prefix(prompt_name)  # rejects unknown names
    return [TextPart((f"{before}\n\n" if before else "") + item.prompt + (f"\n\n{added}" if added else ""))]


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
