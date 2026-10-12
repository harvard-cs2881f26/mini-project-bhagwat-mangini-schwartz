"""Text benchmark loader (Fig. 4; data/text/items.jsonl).

Inputs and answers are loaded by two separate functions, as for video: `load_text_items` returns
only what a model may see (the verbatim prompt), and `load_text_answers` returns the answer key
for the scorer. The request builder only ever receives `TextItem`s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from asimov.bench.text_score import QUESTIONS, response_key
from asimov.data.rows import DATA_DIR, read_answer_json, read_rows


@dataclass(frozen=True)
class TextItem:
    id: str
    prompt: str  # the paper's full eval prompt: scenario, instruction, Q1-Q4, output format


def load_text_items(data_dir: Path = DATA_DIR) -> list[TextItem]:
    return [TextItem(id=r["id"], prompt=r["prompt"]) for r in read_rows("text", data_dir)]


def parse_answer_raw(raw: str) -> dict[str, str]:
    """'```json{"latent_risk_response": "D", ...}```' -> {"latent_risk": "D", ...}"""
    d = read_answer_json(raw)
    return {q: d[response_key(q)] for q in QUESTIONS}


def load_text_answers(data_dir: Path = DATA_DIR) -> dict[str, dict[str, str]]:
    """Answer key, for the scorer only. Never pass this to the request builder."""
    return {r["id"]: parse_answer_raw(r["answer_raw"]) for r in read_rows("text", data_dir)}
