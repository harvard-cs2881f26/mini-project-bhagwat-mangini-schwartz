"""Calibration of the decision models' option probabilities, from fields already saved under results/.

A decision model returns a probability per option for each question (`probs` in responses.jsonl).
Its confidence for a question is the probability it put on the option it answered (`pred` in
per_item; ties go to the service's own answer, so this is not always the largest probability).
The outcome is the strict `correct` flag against the answer key. Questions with no probabilities
(a refusal) or no readable answer are left out and shown as a smaller `n`, never scored as
confidence 0. Formulas come from `asimov.stats` (Brier; ECE with 10 equal-width bins, WORKLOG
2026-10-12 "Statistics methods"). LLM rows save no probabilities, so they get no calibration.
Rule and alternatives: WORKLOG 2026-10-15, "Calibration of the decision models".
"""

from __future__ import annotations

from dataclasses import dataclass

from asimov.bench.text_score import QUESTIONS
from asimov.collect.load import ModelResult
from asimov.stats import brier, ece

QUESTION_TITLES = {
    "latent_risk": "Q1 latent risk type",
    "latent_severity": "Q2 latent severity",
    "effect": "Q3 action effect",
    "activated_risk": "Q4 activated risk",
}


@dataclass(frozen=True)
class Calibration:
    question: str
    n: int  # questions with a probability for the answer given
    n_total: int  # items the model scored
    confidence: float | None  # mean probability on the answer given, over the n
    accuracy: float | None  # share right, over the same n
    brier: float | None  # 0 is perfect
    ece: float | None  # 0 is perfect

    @property
    def gap(self) -> float | None:
        """Mean confidence minus accuracy: above 0 the model is overconfident, below 0 underconfident."""
        return None if self.confidence is None or self.accuracy is None else self.confidence - self.accuracy


def confidences(model: ModelResult, question: str) -> list[tuple[float, bool]]:
    """(probability on the answer given, right?) per item that has both, in item-id order."""
    pairs = []
    for item_id in sorted(model.per_item):
        outcome = model.per_item[item_id].get(question) or {}
        dist = (model.probs.get(item_id) or {}).get(question) or {}
        pred = outcome.get("pred")
        if pred is None or pred not in dist:
            continue
        pairs.append((float(dist[pred]), bool(outcome["correct"])))
    return pairs


def calibration(model: ModelResult, question: str) -> Calibration:
    pairs = confidences(model, question)
    if not pairs:
        return Calibration(question, 0, len(model.per_item), None, None, None, None)
    probs, outcomes = [p for p, _ in pairs], [int(ok) for _, ok in pairs]
    return Calibration(
        question,
        len(pairs),
        len(model.per_item),
        sum(probs) / len(pairs),
        sum(outcomes) / len(pairs),
        brier(probs, outcomes),
        ece(probs, outcomes),
    )


def calibrations(model: ModelResult) -> list[Calibration]:
    """One row per question; empty when the model saved no probabilities (an LLM)."""
    return [calibration(model, q) for q in QUESTIONS] if model.probs else []


def calibration_columns(model: ModelResult) -> dict[str, float | int | None]:
    """index.csv columns: `<question>_brier`, `<question>_ece`, `<question>_n_calibrated`; empty for an LLM."""
    out: dict[str, float | int | None] = {}
    for c in calibrations(model):
        out[f"{c.question}_brier"] = c.brier
        out[f"{c.question}_ece"] = c.ece
        out[f"{c.question}_n_calibrated"] = c.n
    return out
