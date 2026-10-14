"""Part 2 prompt experiments (EXPLORATORY, WORKLOG 2026-10-14, "Part 2 prompt experiments"): an LLM run with a hint
prompt variant against the same LLM alone (verbatim prompt) on the same items. Pure; scripts/prompt_experiments.py
does the I/O.

- kind `hints` (experiments/text_jev_hints): every answer comes from the variant run. Its prompt needs the first
  model's picks, so the system's cost and latency add the first model's call (sequential calls).
- kind `q2_q4` (experiments/text_q2_q4_jev): the composed system takes Q1 from the first model and Q2-Q4 from the
  variant run; cost and latency = both calls, sequential.

Per question: Wilson CIs, the paired bootstrap difference and exact McNemar; mean of 4: bootstrap CI and the paired
bootstrap difference (src/asimov/stats.py, via asimov.cascade).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from asimov.bench.text_score import QUESTIONS
from asimov.cascade import METRICS, Accuracy, ItemResult, RunModel, System, combined, item_score, summarise
from asimov.stats import mcnemar_exact, paired_bootstrap_diff

Kind = Literal["hints", "q2_q4"]
FROM_FIRST: dict[str, tuple[bool, ...]] = {"hints": (False,) * len(QUESTIONS), "q2_q4": (True, False, False, False)}


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Kind
    baseline: RunModel  # the LLM alone, verbatim prompt
    variant: RunModel  # the same LLM with the hint prompt variant


class ExperimentConfig(BaseModel):
    """configs/analysis/prompt_experiments.yaml (unknown keys fail)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str  # printed in bold under the title (EXPLORATORY)
    first: RunModel  # the model whose saved picks the variants show (Jev)
    comparisons: list[Comparison]


@dataclass(frozen=True)
class MetricRow:
    metric: str  # a question, or MEAN
    baseline: Accuracy
    system: Accuracy
    diff: Accuracy  # system minus baseline, paired bootstrap over items
    mcnemar_p: float | None  # exact McNemar, per question; None for the mean of 4
    baseline_only: int | None  # items the baseline got right and the system wrong (per question)
    system_only: int | None


@dataclass(frozen=True)
class Result:
    n: int
    baseline_cost_per_1000: float
    system_cost_per_1000: float
    baseline_p50: float | None
    system_p50: float | None
    first: dict[str, Accuracy]  # the first model alone, for reference
    first_cost_per_1000: float
    first_p50: float | None
    rows: list[MetricRow]


def system_items(kind: str, first: System, variant: System) -> dict[str, ItemResult]:
    """The system's result per item: both calls made, each question from the first model or the variant run."""
    return {i: combined(first.items[i], v, FROM_FIRST[kind]) for i, v in variant.items.items()}


def metric_row(metric: str, base: list[ItemResult], system: list[ItemResult], acc: tuple[dict, dict]) -> MetricRow:
    a = [item_score(x, metric) for x in system]
    b = [item_score(x, metric) for x in base]
    diff = Accuracy(*paired_bootstrap_diff(a, b))
    if metric not in QUESTIONS:
        return MetricRow(metric, acc[0][metric], acc[1][metric], diff, None, None, None)
    base_only = sum(y == 1 and x == 0 for x, y in zip(a, b, strict=True))
    system_only = sum(x == 1 and y == 0 for x, y in zip(a, b, strict=True))
    return MetricRow(metric, acc[0][metric], acc[1][metric], diff, mcnemar_exact(base_only, system_only), base_only, system_only)


def compare(kind: str, first: System, baseline: System, variant: System) -> Result:
    """The system against the LLM alone on the same items. Stops unless all three cover exactly the same items."""
    ids = sorted(variant.items)
    if set(ids) != set(baseline.items) or not set(ids) <= set(first.items):
        raise ValueError(f"{variant.name} and {baseline.name} must have the same items, all answered by {first.name}")
    items = system_items(kind, first, variant)
    system, base, alone = [items[i] for i in ids], [baseline.items[i] for i in ids], [first.items[i] for i in ids]
    base_cost, base_p50, base_acc = summarise(base, METRICS)
    sys_cost, sys_p50, sys_acc = summarise(system, METRICS)
    first_cost, first_p50, first_acc = summarise(alone, METRICS)
    rows = [metric_row(m, base, system, (base_acc, sys_acc)) for m in METRICS]
    return Result(len(ids), base_cost, sys_cost, base_p50, sys_p50, first_acc, first_cost, first_p50, rows)
