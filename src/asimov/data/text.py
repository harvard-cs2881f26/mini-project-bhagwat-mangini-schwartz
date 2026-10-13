"""Text benchmark loader (Fig. 4; data/text/items.jsonl).

Inputs and answers are loaded by two separate functions, as for video: `load_text_items` returns
only what a model may see (the verbatim prompt), and `load_text_answers` returns the answer key
for the scorer. The request builder only ever receives `TextItem`s.

`parse_prompt` splits a verbatim prompt into scenario, instruction and the four questions with
their lettered options, for decision models (Part 2), which take each question separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from asimov.bench.text_score import OPTIONS, QUESTIONS, response_key
from asimov.data.rows import DATA_DIR, read_answer_json, read_rows


@dataclass(frozen=True)
class TextItem:
    id: str
    prompt: str  # the paper's full eval prompt: scenario, instruction, Q1-Q4, output format


@dataclass(frozen=True)
class Question:
    key: str  # text_score.QUESTIONS, e.g. "latent_severity"
    text: str  # the paper's text for this question (stem, guidance, examples), option lines removed
    options: dict[str, str]  # letter -> the paper's option text, in prompt order


@dataclass(frozen=True)
class ParsedPrompt:
    scenario: str
    instruction: str
    questions: tuple[Question, ...]  # in QUESTIONS order


# Every prompt is the same template (checked on all 319): the scenario and instruction, the four
# question blocks, the JSON format, then the scenario and instruction again.
PROMPT = re.compile(r"SCENARIO:(?P<scenario>.*?)\s+and INSTRUCTION:(?P<instruction>.*?)\n\s*(?P<body>Q1\..*?)"
                    r"\n\s*Output the result in JSON format", re.S)
QUESTION_START = re.compile(r"^\s*Q[1-4]\b", re.M)
OPTION_LINE = re.compile(r"^\s*([A-E])\.\s+(.*\S)\s*$", re.M)


def parse_prompt(prompt: str) -> ParsedPrompt:
    """Verbatim text prompt -> scenario, instruction and the 4 questions with options.

    Raises ValueError if the prompt doesn't follow the dataset's template.
    """
    m = PROMPT.search(prompt)
    if m is None:
        raise ValueError("prompt doesn't match the template (SCENARIO: ... and INSTRUCTION: ... Q1. ...)")
    body = m["body"]
    starts = [s.start() for s in QUESTION_START.finditer(body)]
    if len(starts) != len(QUESTIONS):
        raise ValueError(f"expected {len(QUESTIONS)} questions, found {len(starts)}")
    blocks = [body[a:b] for a, b in zip(starts, [*starts[1:], len(body)], strict=True)]
    questions = tuple(
        Question(key=q, text=OPTION_LINE.sub("", block).strip(), options=dict(OPTION_LINE.findall(block)))
        for q, block in zip(QUESTIONS, blocks, strict=True)
    )
    return ParsedPrompt(scenario=m["scenario"].strip(), instruction=m["instruction"].strip(), questions=questions)


def load_text_items(data_dir: Path = DATA_DIR) -> list[TextItem]:
    return [TextItem(id=r["id"], prompt=r["prompt"]) for r in read_rows("text", data_dir)]


def parse_answer_raw(raw: str) -> dict[str, str]:
    """'```json{"latent_risk_response": "D", ...}```' -> {"latent_risk": "D", ...}"""
    d = read_answer_json(raw)
    return {q: d[response_key(q)] for q in QUESTIONS}


OPTION_LINE = re.compile(r"\n([A-E])\. ([^\n]*)")


def parse_options(prompt: str) -> dict[str, dict[str, str]]:
    """Each question's lettered options from the prompt: {question: {letter: text}}.

    Every prompt lists Q1 (A-E) then Q2, Q3, Q4 (A-D each), in that order (checked on all 319).
    The scorer uses this to read a reply that gives an option's text instead of its letter.
    """
    lines = [(m.group(1), m.group(2).strip()) for m in OPTION_LINE.finditer(prompt)]
    sizes = [len(OPTIONS[q]) for q in QUESTIONS]
    if [letter for letter, _ in lines] != [letter for q in QUESTIONS for letter in OPTIONS[q]]:
        raise ValueError(f"expected option lists of sizes {sizes}, found letters {''.join(x for x, _ in lines)}")
    out, i = {}, 0
    for q, n in zip(QUESTIONS, sizes, strict=True):
        out[q] = dict(lines[i : i + n])
        i += n
    return out


def load_text_options(data_dir: Path = DATA_DIR) -> dict[str, dict[str, dict[str, str]]]:
    """Option text per item and question, for the scorer (prompt content, not answers)."""
    return {r["id"]: parse_options(r["prompt"]) for r in read_rows("text", data_dir)}


def load_text_answers(data_dir: Path = DATA_DIR) -> dict[str, dict[str, str]]:
    """Answer key, for the scorer only. Never pass this to the request builder."""
    return {r["id"]: parse_answer_raw(r["answer_raw"]) for r in read_rows("text", data_dir)}
