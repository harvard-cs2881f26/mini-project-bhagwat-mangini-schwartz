"""Same-size comparison on the text benchmark: ~4B Qwen-family systems against TypeSafe's decision models, from saved
replies. EXPLORATORY: framed after the results were seen. Pure: scripts/same_size.py reads results/ and writes the
tables and figures.

- Per row: accuracy per question (Wilson CI) and mean of 4 (bootstrap CI over items), reusing asimov.cascade; the
  Q1-matched Q2-Q4 accuracies and the unreadable (parse-failure) rate copied from summary.json; p50 latency per item.
- Latency: Modal GPU time (`raw.remote.compute_s`) when every item has it, else the recorded round trip (`latency_s`;
  for a cumulative run that is its 4 sequential calls).
- Paired contrasts, fixed in the config: per question, paired bootstrap difference and exact McNemar, Holm across every
  question test in the family; for the mean of 4, the paired bootstrap difference only.

Choices and alternatives: WORKLOG 2026-10-14, "Same-size comparison: Qwen family vs decision models (exploratory)".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from asimov.bench.text_score import QUESTIONS
from asimov.cascade import MEAN, METRICS, Accuracy, System, accuracy, item_score, system_from
from asimov.collect.load import ModelResult
from asimov.prices import Price
from asimov.stats import holm, mcnemar_exact, paired_bootstrap_diff

COMPUTE = "Modal compute_s"
ROUND_TRIP = "round trip"
ROUND_TRIP_CUMULATIVE = "round trip, 4 sequential calls"
MATCHED_QUESTIONS = QUESTIONS[1:]  # Q2-Q4, scored on the items whose Q1 matched the key
PER_THOUSAND = 1000
SUMMARY_TOLERANCE = 1e-9


class RowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str  # the run name (config `run`), never a folder name
    model: str  # the model id in that run's config
    label: str  # unique; contrasts refer to rows by it
    group: str
    flag: str | None = None  # printed next to the row (e.g. a parse-failure warning)


class ContrastSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    a: str  # row label; differences are a minus b
    b: str
    kind: Literal["pre-registered", "exploratory"]
    changes: str  # what differs between the two rows


class SameSizeConfig(BaseModel):
    """configs/analysis/<name>.yaml: the rows and the contrasts, fixed before looking at the contrast results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str  # printed at the top of every output
    rows: list[RowSpec]
    contrasts: list[ContrastSpec]

    @model_validator(mode="after")
    def known_rows(self) -> SameSizeConfig:
        labels = [r.label for r in self.rows]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            raise ValueError(f"duplicate row labels: {dupes}")
        unknown = sorted({x for c in self.contrasts for x in (c.a, c.b)} - set(labels))
        if unknown:
            raise ValueError(f"contrasts name rows that aren't configured: {unknown}")
        return self


def item_latencies(
    items: Iterable[str], round_trip: dict[str, float], compute: dict[str, float], cumulative: bool
) -> tuple[dict[str, float], str]:
    """Each item's latency and its source: Modal compute_s when every item has it, else the recorded round trip."""
    ids = sorted(items)
    if compute:
        missing = [i for i in ids if i not in compute]
        if missing:
            raise ValueError(f"raw.remote.compute_s missing for {len(missing)} items: {missing[:10]}")
        return {i: compute[i] for i in ids}, COMPUTE
    missing = [i for i in ids if i not in round_trip]
    if missing:
        raise ValueError(f"no saved latency for {len(missing)} items: {missing[:10]}")
    return {i: round_trip[i] for i in ids}, ROUND_TRIP_CUMULATIVE if cumulative else ROUND_TRIP


@dataclass(frozen=True)
class RowResult:
    spec: RowSpec
    system: System  # per-item outcomes, latency from `latency_source`; the contrasts pair these
    n: int
    accuracy: dict[str, Accuracy]  # METRICS
    q1_matched: dict[str, float | None]  # Q2-Q4 on the items whose Q1 matched (summary.json)
    n_q1_matched: int | None
    unreadable_rate: float  # summary parse_failure_rate
    latency_p50: float
    latency_source: str
    params_b: float | None
    active_params_b: float | None
    cost_per_1000: float | None  # $ per 1,000 items; None when self-hosted (GPU time, no token price)


def check_summary(system: System, summary: dict) -> None:
    """Per-question accuracy rebuilt from per_item must equal summary.json's."""
    if len(system.items) != summary.get("n"):
        raise ValueError(f"{system.name}: {len(system.items)} scored items, summary n = {summary.get('n')}")
    for q in QUESTIONS:
        mine = accuracy([item_score(x, q) for x in system.items.values()], q).value
        if abs(mine - summary[f"{q}_accuracy"]) > SUMMARY_TOLERANCE:
            raise ValueError(f"{system.name}: {q} accuracy {mine} from per_item, {summary[f'{q}_accuracy']} in summary")


def row_result(spec: RowSpec, model: ModelResult, price: Price, cumulative: bool) -> RowResult:
    system = system_from(model, price)
    check_summary(system, model.summary)
    latency, source = item_latencies(system.items, model.item_latency, model.item_compute_s, cumulative)
    items = {i: replace(x, latency=latency[i]) for i, x in system.items.items()}
    system = System(spec.label, items, {})
    outcomes = [items[i] for i in sorted(items)]
    s = model.summary
    cost = None if price.self_hosted else sum(x.cost or 0.0 for x in outcomes) / len(outcomes) * PER_THOUSAND
    return RowResult(
        spec=spec,
        system=system,
        n=len(outcomes),
        accuracy={m: accuracy([item_score(x, m) for x in outcomes], m) for m in METRICS},
        q1_matched={q: s.get(f"{q}_accuracy_q1_matched") for q in MATCHED_QUESTIONS},
        n_q1_matched=s.get("n_q1_matched"),
        unreadable_rate=float(s["parse_failure_rate"]),
        latency_p50=float(np.median(list(latency.values()))),
        latency_source=source,
        params_b=price.params_b,
        active_params_b=price.active_params_b,
        cost_per_1000=cost,
    )


@dataclass(frozen=True)
class PairedTest:
    metric: str  # MEAN or a question
    diff: Accuracy  # a minus b, paired bootstrap 95% CI
    a_only: int | None  # items a got right and b wrong (questions only)
    b_only: int | None
    p: float | None  # exact McNemar (questions only)
    p_holm: float | None  # Holm across every question test in the family


@dataclass(frozen=True)
class ContrastResult:
    spec: ContrastSpec
    n: int
    tests: tuple[PairedTest, ...]  # MEAN first, then QUESTIONS


def paired(spec: ContrastSpec, a: System, b: System) -> ContrastResult:
    if set(a.items) != set(b.items):
        only = sorted(set(a.items) ^ set(b.items))
        raise ValueError(f"{a.name} vs {b.name}: the items differ ({len(only)} in one only: {only[:10]})")
    ids = sorted(a.items)
    tests = []
    for m in METRICS:
        xa = [item_score(a.items[i], m) for i in ids]
        xb = [item_score(b.items[i], m) for i in ids]
        diff = Accuracy(*paired_bootstrap_diff(xa, xb))
        if m == MEAN:
            tests.append(PairedTest(m, diff, None, None, None, None))
            continue
        a_only = sum(1 for x, y in zip(xa, xb, strict=True) if x > y)
        b_only = sum(1 for x, y in zip(xa, xb, strict=True) if y > x)
        tests.append(PairedTest(m, diff, a_only, b_only, mcnemar_exact(a_only, b_only), None))
    return ContrastResult(spec, len(ids), tuple(tests))


def contrast_family(specs: Sequence[ContrastSpec], systems: dict[str, System]) -> list[ContrastResult]:
    """Every contrast, with Holm-adjusted McNemar p-values over all their question tests together."""
    results = [paired(c, systems[c.a], systems[c.b]) for c in specs]
    raw = [t.p for r in results for t in r.tests if t.p is not None]
    adjusted = iter(holm(raw))
    return [replace(r, tests=tuple(t if t.p is None else replace(t, p_holm=next(adjusted)) for t in r.tests)) for r in results]
