"""Ours vs the paper (configs/paper/claims.yaml): one row per claim, plus the three headline checks.

Numbers and mechanical checks only (is the paper value inside our CI, gaps with paired CIs);
whether the reproduction holds is the owner's call (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asimov.claims import Claim, Claims, Headline, as_fraction, is_stand_in, model_key
from asimov.collect.load import ModelResult, RunData
from asimov.collect.metrics import Estimate, Paired, correct_by_question, estimate, paired
from asimov.stats import holm, paired_bootstrap_diff

# Fig 4b questions vs Fig 4a questions, for the "actions are harder" check (names from bench/text_score.py)
ACTION_QUESTIONS = ("effect", "activated_risk")
LATENT_QUESTIONS = ("latent_risk", "latent_severity")
UNCHECKED_KINDS = {"bound": "a bound, not a point value", "baseline": "derived from the dataset's counts, no model"}


def find_model(run: RunData, paper_name: str | None) -> ModelResult | None:
    key = model_key(paper_name or "")
    return next((m for m in run.models if model_key(m.name) == key), None)


@dataclass(frozen=True)
class CheckRow:
    claim: Claim
    paper: float  # on our scale (fractions, seconds)
    ours: Estimate | None
    our_models: tuple[str, ...]  # our config names behind `ours`
    status: str  # "checked", or why not

    @property
    def diff(self) -> float | None:
        return None if self.ours is None else self.ours.value - self.paper

    @property
    def paper_in_ci(self) -> bool | None:
        return None if self.ours is None or self.claim.kind != "value" else self.ours.contains(self.paper)

    @property
    def stand_in(self) -> bool:
        return any(is_stand_in(n) for n in self.our_models)


def check_value(run: RunData, c: Claim) -> tuple[Estimate | None, tuple[str, ...], str]:
    m = find_model(run, c.model)
    if m is None:
        return None, (), "model not in this run"
    est = estimate(m, run.benchmark, c.metric)
    return est, (m.name,), "checked" if est else "metric not in our summary"


def check_group(run: RunData, c: Claim) -> tuple[Estimate | None, tuple[str, ...], str]:
    """mean / max / min over the claim's models, from our summaries (no CI)."""
    found = [m for m in (find_model(run, name) for name in c.models) if m is not None]
    values = [m.summary[c.metric] for m in found if m.summary.get(c.metric) is not None]
    if not values:
        return None, (), "models or metric not in this run"
    reduce = {"mean": np.mean, "max": max, "min": min}[c.kind]
    status = "checked" if len(values) == len(c.models) else f"only {len(values)} of {len(c.models)} models in this run"
    return Estimate(float(reduce(values)), None, None, len(values)), tuple(m.name for m in found), status


def check_gap(run: RunData, c: Claim) -> tuple[Estimate | None, tuple[str, ...], str]:
    """versus minus model, paired over items when the metric is a per-question accuracy."""
    if c.versus is None:
        return None, (), "not checked by code (see note)"
    small, large = find_model(run, c.model), find_model(run, c.versus)
    if small is None or large is None:
        return None, (), "model not in this run"
    names = (large.name, small.name)
    question = c.metric.removesuffix("_accuracy")
    by_q_large, by_q_small = correct_by_question(large, run.benchmark), correct_by_question(small, run.benchmark)
    if c.metric.endswith("_accuracy") and question in by_q_large and question in by_q_small:
        p = paired(by_q_large[question], by_q_small[question])
        if p:
            return Estimate(p.diff, p.lo, p.hi, p.n), names, "checked"
    a, b = large.summary.get(c.metric), small.summary.get(c.metric)
    if a is None or b is None:
        return None, names, "metric not in our summary"
    return Estimate(a - b, None, None, 0), names, "checked (no paired CI for this metric)"


def check_claim(run: RunData, c: Claim) -> CheckRow:
    if c.kind in UNCHECKED_KINDS:
        result = (None, (), UNCHECKED_KINDS[c.kind])
    elif c.kind == "value":
        result = check_value(run, c)
    elif c.kind == "gap":
        result = check_gap(run, c)
    else:
        result = check_group(run, c)
    return CheckRow(c, as_fraction(c), *result)


def paper_check(run: RunData, claims: Claims) -> list[CheckRow]:
    return [check_claim(run, c) for c in claims.claims if c.benchmark == run.benchmark]


@dataclass(frozen=True)
class CloseRow:
    model: str  # our name
    n_checked: int
    n_inside: int  # paper value inside our 95% CI
    mean_abs_diff: float


def close_to_paper(rows: list[CheckRow], h: Headline) -> list[CloseRow]:
    out = []
    for paper_name in h.models:
        mine = [r for r in rows if r.claim.kind == "value" and r.ours and model_key(r.claim.model or "") == model_key(paper_name)]
        if mine:
            inside = sum(bool(r.paper_in_ci) for r in mine)
            mad = float(np.mean([abs(r.diff or 0.0) for r in mine]))
            out.append(CloseRow(mine[0].our_models[0], len(mine), inside, mad))
    return out


@dataclass(frozen=True)
class GapRow:
    large: str  # our names
    small: str
    question: str
    result: Paired
    p_holm: float


def size_gap(run: RunData, h: Headline) -> list[GapRow]:
    """Per (large, small) pair and question: large minus small, paired; Holm across all rows."""
    raw = []
    for large_name, small_name in h.pairs:
        large, small = find_model(run, large_name), find_model(run, small_name)
        if large is None or small is None:
            continue
        by_large, by_small = correct_by_question(large, run.benchmark), correct_by_question(small, run.benchmark)
        for q in by_large:
            p = paired(by_large[q], by_small.get(q, {}))
            if p:
                raw.append((large.name, small.name, q, p))
    adjusted = holm([p.p for *_, p in raw])
    return [GapRow(a, b, q, p, adj) for (a, b, q, p), adj in zip(raw, adjusted, strict=True)]


@dataclass(frozen=True)
class HarderRow:
    model: str
    action: float  # mean accuracy over ACTION_QUESTIONS
    latent: float
    diff: float  # action - latent, per item then averaged
    lo: float
    hi: float


def actions_harder(run: RunData) -> list[HarderRow]:
    """Per model: per item, mean correct on the action questions minus on the latent ones (paired bootstrap)."""
    out = []
    for m in run.models:
        by_q = correct_by_question(m, run.benchmark)
        if not all(q in by_q for q in ACTION_QUESTIONS + LATENT_QUESTIONS):
            continue
        items = sorted(by_q[ACTION_QUESTIONS[0]])
        action = [np.mean([by_q[q][i] for q in ACTION_QUESTIONS]) for i in items]
        latent = [np.mean([by_q[q][i] for q in LATENT_QUESTIONS]) for i in items]
        diff, lo, hi = paired_bootstrap_diff(action, latent)
        out.append(HarderRow(m.name, float(np.mean(action)), float(np.mean(latent)), diff, lo, hi))
    return out
