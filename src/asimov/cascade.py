"""Part 2 designs B and D on the text benchmark, simulated from saved replies (docs/text-extension-notes.md,
"Design options"). Pure: the script (scripts/cascade.py) reads results/ and writes the tables and figures.

- D, route by confidence: the decision model answers every item; items whose confidence is below a threshold are
  also sent to the LLM, whose answers replace the decision model's. One LLM call answers all 4 questions, so
  routing is per item. The threshold sweeps every distinct confidence, giving the whole curve.
- Perfect router: the upper bound. It sends an item on only when the LLM gets strictly more of the metric right,
  largest gain first.
- B, split by question: the decision model answers some questions, the LLM the rest; all 16 splits. The LLM's one
  call still answers all 4, so any split using the LLM costs both calls.

Choices (confidence rule, targets, headline split): WORKLOG 2026-10-13, "Part 2 design analysis, B and D".
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from asimov.bench.text_score import QUESTIONS
from asimov.collect.load import ModelResult
from asimov.prices import Price, call_cost
from asimov.stats import bootstrap_mean_ci, paired_bootstrap_diff, wilson_ci

K = TypeVar("K", str, int)  # an item key: an item id, or an index into a fixed item order
MEAN = "mean"  # mean accuracy over the 4 questions, per item
METRICS = (MEAN, *QUESTIONS)
PER_THOUSAND = 1000


class RunModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str  # the run name (config `run`), never a folder name
    model: str  # the model id in that run's config


class DesignConfig(BaseModel):
    """configs/analysis/<name>.yaml: which saved runs the designs combine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    escalation: list[RunModel]  # LLMs that get the routed items (D) or the other questions (B); each paired with every `first`
    first: list[RunModel]  # decision models that answer first
    confidence: Literal["min_top_probability"]
    fixed_fractions: tuple[float, ...]  # D's table: share of items sent on, fixed before looking at results
    split_headline: tuple[str, ...]  # B's pre-registered split: the questions the decision model answers
    label: str | None = None  # printed in bold under the tables' title and in each figure's title (e.g. EXPLORATORY)

    @field_validator("fixed_fractions")
    @classmethod
    def fractions(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0 <= f <= 1 for f in v):
            raise ValueError(f"fixed_fractions must be in [0, 1], got {v}")
        return v

    @field_validator("split_headline")
    @classmethod
    def known_questions(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [q for q in v if q not in QUESTIONS]
        if unknown:
            raise ValueError(f"unknown questions {unknown}; use {QUESTIONS}")
        return tuple(q for q in QUESTIONS if q in v)


@dataclass(frozen=True)
class ItemResult:
    correct: tuple[bool, ...]  # per question, QUESTIONS order
    cost: float | None  # $; None for a self-hosted model (no $ price)
    latency: float | None  # s


@dataclass(frozen=True)
class System:
    name: str
    items: dict[str, ItemResult]
    confidence: dict[str, float]  # item -> confidence; empty for a system that never answers first


def item_confidence(probs: dict | None) -> float:
    """The lowest top-option probability across the 4 questions; 0 when any question has no probabilities."""
    tops = [max(((probs or {}).get(q) or {"": 0.0}).values()) for q in QUESTIONS]
    return float(min(tops))


CONFIDENCE_RULES: dict[str, Callable[[dict | None], float]] = {"min_top_probability": item_confidence}


def system_from(model: ModelResult, price: Price, confidence_rule: str | None = None) -> System:
    """One model's scored replies as a System. Stops on failed calls: their items would be silently missing."""
    errors = model.summary.get("errors") or 0
    if errors:
        raise ValueError(f"{model.name}: {errors} failed calls; rerun scripts/run.py on its config to retry them first")
    for field, found in (("usage", model.usage), ("latency", model.item_latency)):
        missing = sorted(set(model.per_item) - set(found))
        if missing:  # it would count as $0 or drop out of the latency median
            raise ValueError(f"{model.name}: no saved {field} for {len(missing)} scored items: {missing[:10]}")
    items = {
        i: ItemResult(tuple(bool(o[q]["correct"]) for q in QUESTIONS), call_cost(model.usage[i], price), model.item_latency[i])
        for i, o in model.per_item.items()
    }
    rule = CONFIDENCE_RULES[confidence_rule] if confidence_rule else None
    confidence = {i: rule(model.probs.get(i)) for i in items} if rule else {}
    return System(model.name, items, confidence)


def check_pair(first: System, escalate: System) -> None:
    missing = sorted(set(first.items) - set(escalate.items))
    if missing:
        raise ValueError(
            f"{len(missing)} {first.name} items have no {escalate.name} reply: {missing[:10]}; rerun the LLM's config"
        )


def check_routable(first: System, escalate: System) -> None:
    """D needs a confidence for every item the decision model answered."""
    check_pair(first, escalate)
    if set(first.confidence) != set(first.items):
        raise ValueError(f"{first.name}: no confidence for some items (is it a decision model?)")


def item_score(item: ItemResult, metric: str) -> float:
    return sum(item.correct) / len(QUESTIONS) if metric == MEAN else float(item.correct[QUESTIONS.index(metric)])


@dataclass(frozen=True)
class Accuracy:
    value: float
    lo: float  # 95% CI
    hi: float


def accuracy(scores: Sequence[float], metric: str) -> Accuracy:
    """Wilson CI for one question; percentile bootstrap over items for the mean over the 4 (src/asimov/stats.py)."""
    if metric == MEAN:
        return Accuracy(*bootstrap_mean_ci(scores))
    k, n = int(sum(scores)), len(scores)
    return Accuracy(k / n, *wilson_ci(k, n))


def add(a: float | None, b: float | None) -> float | None:
    return None if a is None and b is None else (a or 0.0) + (b or 0.0)


def combined(a: ItemResult, b: ItemResult, from_first: tuple[bool, ...]) -> ItemResult:
    """Both calls made: each question's answer from the decision model (True) or the LLM (False)."""
    correct = tuple(x if keep else y for x, y, keep in zip(a.correct, b.correct, from_first, strict=True))
    latency = None if a.latency is None or b.latency is None else a.latency + b.latency  # sequential calls
    return ItemResult(correct, add(a.cost, b.cost), latency)


@dataclass(frozen=True)
class Point:
    """One routing on D's curve or the perfect router's frontier."""

    threshold: float | None  # sent on when confidence < threshold; None on the perfect router or at a fixed fraction
    n: int
    n_sent: int
    cost_per_1000: float  # $ per 1,000 items; a self-hosted model's own calls add $0
    latency_p50: float | None  # s per item, both calls when sent on
    accuracy: dict[str, Accuracy]


def summarise(items: list[ItemResult], metrics: Sequence[str]) -> tuple[float, float | None, dict[str, Accuracy]]:
    cost = sum(i.cost or 0.0 for i in items) / len(items) * PER_THOUSAND
    latencies = [i.latency for i in items if i.latency is not None]
    p50 = float(np.median(latencies)) if latencies else None
    return cost, p50, {m: accuracy([item_score(i, m) for i in items], m) for m in metrics}


def routed(first: System, escalate: System, sent: set[str], threshold: float | None, metrics: Sequence[str]) -> Point:
    everything_llm = (False,) * len(QUESTIONS)
    items = [
        combined(first.items[i], escalate.items[i], everything_llm) if i in sent else first.items[i] for i in sorted(first.items)
    ]
    return Point(threshold, len(items), len(sent), *summarise(items, metrics))


def cascade_curve(first: System, escalate: System, metrics: Sequence[str] = METRICS) -> list[Point]:
    """D: one point per distinct confidence (items below it are sent on), then everything sent on."""
    check_routable(first, escalate)
    thresholds = [*sorted(set(first.confidence.values())), float("inf")]
    return [routed(first, escalate, {i for i, c in first.confidence.items() if c < t}, t, metrics) for t in thresholds]


def at_fraction(first: System, escalate: System, fraction: float, metrics: Sequence[str] = METRICS) -> Point:
    """D at a fixed share of items sent on: exactly round(fraction x n) items, lowest confidence first, ties by item id
    (a confidence threshold alone can't split tied items)."""
    check_routable(first, escalate)
    k = math.floor(fraction * len(first.items) + 0.5)
    order = sorted(first.items, key=lambda i: (first.confidence[i], i))
    return routed(first, escalate, set(order[:k]), None, metrics)


def perfect_router(first: System, escalate: System, metric: str) -> list[Point]:
    """The upper bound for one metric at each number of items sent on: send items on only when the LLM scores strictly
    higher, largest gain first (then the cheaper LLM call, then item id). Ordered by items, not $: a bound at equal
    items sent, which the figures draw against $."""
    check_pair(first, escalate)
    gain = {i: item_score(escalate.items[i], metric) - item_score(a, metric) for i, a in first.items.items()}
    order = gain_order(gain, {i: escalate.items[i].cost or 0.0 for i in gain})
    return [routed(first, escalate, set(order[:k]), None, (metric,)) for k in range(len(order) + 1)]


def gain_order(gain: Mapping[K, float], llm_cost: Mapping[K, float]) -> list[K]:
    """The perfect router's order: items where the LLM scores strictly higher, largest gain first, then the cheaper
    LLM call, then item key."""
    return sorted((i for i, g in gain.items() if g > 0), key=lambda i: (-gain[i], llm_cost[i], i))


@dataclass(frozen=True)
class SplitRow:
    decision_questions: tuple[str, ...]  # answered by the decision model; the LLM answers the rest
    headline: bool  # the pre-registered split
    cost_per_1000: float
    latency_p50: float | None
    accuracy: dict[str, Accuracy]
    diff_vs_llm: Accuracy | None  # headline only: mean over the 4 questions minus the LLM alone's, paired bootstrap


def split_row(first: System, escalate: System, from_first: tuple[bool, ...], headline: tuple[str, ...]) -> SplitRow:
    def item(i: str) -> ItemResult:  # a system alone makes only its own call
        if all(from_first):
            return first.items[i]
        return escalate.items[i] if not any(from_first) else combined(first.items[i], escalate.items[i], from_first)

    ids = sorted(first.items)
    items = [item(i) for i in ids]
    questions = tuple(q for q, keep in zip(QUESTIONS, from_first, strict=True) if keep)
    diff = None
    if questions == headline:
        llm = [item_score(escalate.items[i], MEAN) for i in ids]
        diff = Accuracy(*paired_bootstrap_diff([item_score(x, MEAN) for x in items], llm))
    return SplitRow(questions, questions == headline, *summarise(items, METRICS), diff)


def split_table(first: System, escalate: System, headline: tuple[str, ...]) -> list[SplitRow]:
    """B: all 16 splits, from everything by the decision model to everything by the LLM."""
    check_pair(first, escalate)
    return [split_row(first, escalate, mask, headline) for mask in itertools.product((True, False), repeat=len(QUESTIONS))]
