"""Router search (exploratory): combinations of decision models and LLMs on the text benchmark, simulated from saved
per-item correctness, `probs`, usage and latency. Pure: scripts/router_search.py reads results/ and writes the tables.

Every candidate is a rule fitted on training items only (a confidence threshold from the training items'
confidences, or a per-question choice from their correctness), then applied to every item. Choosing among candidates
also uses training items only: 5-fold cross-validation over items, repeated with fixed seeds. WORKLOG 2026-10-13,
"Router search (exploratory)".

Families:
1. single: one system alone.
2. d_cascade: the decision model answers; the lowest-confidence share of items also goes to the LLM, whose answers
   replace it (src/asimov/cascade.py's design D, with the threshold fitted on training items).
3. agreement: two first-stage systems answer every item (in parallel); items where they disagree (on any question,
   or on Q1 only) go to a strong LLM. Otherwise the second system's answers are kept.
4. three_stage: decision model; low-confidence items to a cheap LLM; of those, items where the cheap LLM disagrees
   with the decision model to a strong LLM.
5. per_question_mix: design D, but on items sent on, each question keeps the decision model's answer when it beat
   the LLM's on the training items sent on (design B within D). One LLM call still answers all 4.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from asimov.bench.text_score import QUESTIONS
from asimov.cascade import MEAN, PER_THOUSAND, Accuracy, accuracy, system_from
from asimov.collect.load import ModelResult
from asimov.prices import Price

Rule = Literal["any", "q1"]
CapKind = Literal["cost", "latency"]
FAMILIES = ("single", "d_cascade", "agreement", "three_stage", "per_question_mix")
UNANSWERED = ""  # a missing answer never agrees with anything


class PoolEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str  # run name (config `run`)
    model: str  # model id in that run
    label: str  # unique name in the tables


class SearchConfig(BaseModel):
    """configs/analysis/router_search.yaml (unknown keys fail)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: list[PoolEntry]  # first-stage decision models (need `probs`)
    cumulative_runs: str  # fnmatch pattern on run names: their decision models join the pool when errors == 0
    llms: list[PoolEntry]
    cheap: list[str]  # labels from llms: agreement partners and three_stage's middle stage
    strong: list[str]  # labels from llms: agreement and three_stage escalation targets
    confidence: Literal["min_top_probability"]
    fractions: tuple[float, ...]  # share of training items sent on (threshold = that quantile of their confidence)
    agreement_rules: tuple[Rule, ...]
    budgets_per_1000: tuple[float, ...]  # $ per 1,000 items
    latency_caps_s: tuple[float, ...]  # p50 latency per item
    folds: int
    seeds: tuple[int, ...]  # one repeat per seed

    @field_validator("fractions")
    @classmethod
    def strictly_inside(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0 < f < 1 for f in v):
            raise ValueError(f"fractions must be in (0, 1); 0 and 1 are the single systems: {v}")
        return v

    @model_validator(mode="after")
    def tiers_are_llms(self) -> SearchConfig:
        labels = [e.label for e in [*self.decision, *self.llms]]
        if len(set(labels)) != len(labels):
            raise ValueError(f"labels must be unique: {labels}")
        unknown = sorted(set(self.cheap + self.strong) - {e.label for e in self.llms})
        if unknown:
            raise ValueError(f"cheap/strong labels not in llms: {unknown}")
        return self


@dataclass(frozen=True, eq=False)
class Member:
    """One system's saved replies as arrays over a fixed item order."""

    label: str
    correct: np.ndarray  # (n, 4) bool, QUESTIONS order
    pred: np.ndarray  # (n, 4) letters; UNANSWERED when missing
    cost: np.ndarray  # (n,) $ per item; 0 for a self-hosted model
    latency: np.ndarray  # (n,) s
    confidence: np.ndarray | None  # (n,) decision models only
    self_hosted: bool


def member_from(label: str, model: ModelResult, price: Price, ids: Sequence[str], rule: str | None) -> Member:
    """Validated by cascade.system_from (no failed calls, usage and latency for every item)."""
    system = system_from(model, price, rule)
    missing = sorted(set(ids) - set(system.items))
    if missing:
        raise ValueError(f"{label}: no reply for {len(missing)} items: {missing[:10]}; every pool member needs all items")
    items = [system.items[i] for i in ids]
    pred = [[model.per_item[i][q].get("pred") or UNANSWERED for q in QUESTIONS] for i in ids]
    return Member(
        label,
        np.array([it.correct for it in items], dtype=bool),
        np.array(pred, dtype=object),
        np.array([it.cost or 0.0 for it in items]),
        np.array([it.latency for it in items], dtype=float),
        np.array([system.confidence[i] for i in ids]) if rule else None,
        bool(price.self_hosted),
    )


@dataclass(frozen=True)
class Outcome:
    correct: np.ndarray  # (n, 4)
    cost: np.ndarray  # (n,)
    latency: np.ndarray  # (n,)


@dataclass(frozen=True)
class Candidate:
    family: str
    name: str  # its full definition
    self_hosted: bool  # uses a self-hosted model, whose cost is not in $
    fit: Callable[[np.ndarray], Outcome]  # training item indices -> outcome on every item


def alone(m: Member) -> Outcome:
    return Outcome(m.correct, m.cost, m.latency)


def sent_below(confidence: np.ndarray, train: np.ndarray, fraction: float) -> np.ndarray:
    """Items whose confidence is below the `fraction` quantile of the training items' confidences."""
    return confidence < np.quantile(confidence[train], fraction)


def escalate(base: Outcome, to: Member, sent: np.ndarray, keep: np.ndarray | None = None) -> Outcome:
    """Sent items also call `to` (sequentially); its answers replace base's except on `keep` questions."""
    take = sent[:, None] & ~(np.zeros(len(QUESTIONS), bool) if keep is None else keep)[None, :]
    return Outcome(np.where(take, to.correct, base.correct), base.cost + to.cost * sent, base.latency + to.latency * sent)


def disagree(a: np.ndarray, b: np.ndarray, rule: Rule) -> np.ndarray:
    """Items where two systems' answers differ: on any question, or on Q1 only. Unanswered never agrees."""
    cols = slice(None) if rule == "any" else slice(0, 1)
    differ = (a[:, cols] != b[:, cols]) | (a[:, cols] == UNANSWERED) | (b[:, cols] == UNANSWERED)
    return differ.any(axis=1)


def needs_confidence(m: Member) -> np.ndarray:
    if m.confidence is None:
        raise ValueError(f"{m.label} has no confidence: it can't route (is it a decision model?)")
    return m.confidence


def single(m: Member) -> Candidate:
    return Candidate("single", m.label, m.self_hosted, lambda train: alone(m))


def d_cascade(dec: Member, llm: Member, fraction: float) -> Candidate:
    conf = needs_confidence(dec)
    name = f"{dec.label} -> {llm.label} (lowest {fraction:.0%} by confidence)"
    return Candidate(
        "d_cascade", name, dec.self_hosted or llm.self_hosted, lambda t: escalate(alone(dec), llm, sent_below(conf, t, fraction))
    )


def per_question_mix(dec: Member, llm: Member, fraction: float) -> Candidate:
    conf = needs_confidence(dec)

    def fit(train: np.ndarray) -> Outcome:
        sent = sent_below(conf, train, fraction)
        mine = train[sent[train]]
        keep = dec.correct[mine].mean(axis=0) > llm.correct[mine].mean(axis=0) if len(mine) else None  # ties: the LLM
        return escalate(alone(dec), llm, sent, keep)

    name = f"{dec.label} -> {llm.label} (lowest {fraction:.0%} by confidence; per question, the better on training)"
    return Candidate("per_question_mix", name, dec.self_hosted or llm.self_hosted, fit)


def agreement(a: Member, b: Member, strong: Member, rule: Rule) -> Candidate:
    """a and b answer in parallel (latency: the slower); b's answers are kept where they agree under `rule`."""
    first = Outcome(b.correct, a.cost + b.cost, np.maximum(a.latency, b.latency))
    sent = disagree(a.pred, b.pred, rule)
    name = f"{a.label} + {b.label} agree ({rule}) else {strong.label}"
    hosted = a.self_hosted or b.self_hosted or strong.self_hosted
    return Candidate("agreement", name, hosted, lambda train: escalate(first, strong, sent))


def three_stage(dec: Member, cheap: Member, strong: Member, fraction: float, rule: Rule) -> Candidate:
    conf = needs_confidence(dec)
    differ = disagree(dec.pred, cheap.pred, rule)

    def fit(train: np.ndarray) -> Outcome:
        low = sent_below(conf, train, fraction)
        return escalate(escalate(alone(dec), cheap, low), strong, low & differ)

    name = f"{dec.label} -> {cheap.label} (lowest {fraction:.0%}) -> {strong.label} if they disagree ({rule})"
    return Candidate("three_stage", name, dec.self_hosted or cheap.self_hosted or strong.self_hosted, fit)


def candidates(
    decisions: Sequence[Member],
    llms: Sequence[Member],
    cheap: Sequence[Member],
    strong: Sequence[Member],
    fractions: Sequence[float],
    rules: Sequence[Rule],
) -> list[Candidate]:
    """The finite search space. Agreement pairs: decision + decision (unordered for `any`, where the kept answers are
    the same; ordered for `q1`), and decision + cheap LLM (the LLM's answers kept)."""
    out = [single(m) for m in [*decisions, *llms]]
    out += [d_cascade(d, m, f) for d in decisions for m in llms for f in fractions]
    for rule in rules:
        pairs = [(a, b) for i, a in enumerate(decisions) for j, b in enumerate(decisions) if (i < j if rule == "any" else i != j)]
        pairs += [(d, c) for d in decisions for c in cheap]
        out += [agreement(a, b, s, rule) for a, b in pairs for s in strong]
    out += [three_stage(d, c, s, f, r) for r in rules for d in decisions for c in cheap for s in strong for f in fractions]
    out += [per_question_mix(d, m, f) for d in decisions for m in llms for f in fractions]
    return out


# --- metrics and selection ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    accuracy: np.ndarray  # (C,) mean over items of the share of the 4 questions right
    cost: np.ndarray  # (C,) $ per 1,000 items
    latency: np.ndarray  # (C,) p50 s per item


def metrics_on(outcomes: Sequence[Outcome], idx: np.ndarray) -> Metrics:
    score = np.stack([o.correct[idx].mean(axis=1) for o in outcomes])
    cost = np.stack([o.cost[idx] for o in outcomes])
    latency = np.stack([o.latency[idx] for o in outcomes])
    return Metrics(score.mean(axis=1), cost.mean(axis=1) * PER_THOUSAND, np.median(latency, axis=1))


def select(m: Metrics, kind: CapKind, cap: float) -> int | None:
    """Highest accuracy with cost (or p50 latency) within the cap; ties: cheaper, then faster, then listed first."""
    capped = m.cost if kind == "cost" else m.latency
    ok = np.flatnonzero(capped <= cap)
    if not len(ok):
        return None
    order = np.lexsort((ok, m.latency[ok], m.cost[ok], -m.accuracy[ok]))
    return int(ok[order[0]])


def pareto(acc: Sequence[float], x: Sequence[float]) -> list[int]:
    """Indices of points no other point beats on both (higher accuracy, lower x), by increasing x."""
    order = sorted(range(len(acc)), key=lambda i: (x[i], -acc[i], i))
    out, best = [], -np.inf
    for i in order:
        if acc[i] > best:
            out.append(i)
            best = acc[i]
    return out


# --- cross-validation -----------------------------------------------------------------------------------------


def fold_ids(n: int, folds: int, seed: int) -> np.ndarray:
    """Fold of each item: a shuffle with default_rng(seed), then round-robin (no stratification)."""
    out = np.empty(n, dtype=int)
    out[np.random.default_rng(seed).permutation(n)] = np.arange(n) % folds
    return out


@dataclass(frozen=True)
class Pick:
    seed: int
    fold: int
    candidate: int | None  # index into the candidate list; None when nothing fits the cap


@dataclass
class HeldOut:
    """One cap's held-out results: each item scored by the candidate picked without it, per repeat."""

    picks: list[Pick] = field(default_factory=list)
    correct: list[np.ndarray] = field(default_factory=list)  # per repeat, (n, 4) float; NaN where nothing fit
    cost: list[np.ndarray] = field(default_factory=list)
    latency: list[np.ndarray] = field(default_factory=list)


def cross_validate(
    cands: Sequence[Candidate], n: int, folds: int, seeds: Sequence[int], caps: Mapping[str, Sequence[float]]
) -> dict[tuple[str, float], HeldOut]:
    """For each repeat and fold: fit every candidate on the training folds, pick per cap on them, and record the pick's
    outcome on the held-out fold."""
    out = {(kind, cap): HeldOut() for kind, values in caps.items() for cap in values}
    for seed in seeds:
        fold = fold_ids(n, folds, seed)
        for h in out.values():
            h.correct.append(np.full((n, len(QUESTIONS)), np.nan))
            h.cost.append(np.full(n, np.nan))
            h.latency.append(np.full(n, np.nan))
        for k in range(folds):
            train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            outcomes = [c.fit(train) for c in cands]
            on_train = metrics_on(outcomes, train)
            for (kind, cap), h in out.items():
                pick = select(on_train, kind, cap)  # pyright: ignore[reportArgumentType]
                h.picks.append(Pick(seed, k, pick))
                if pick is not None:
                    o = outcomes[pick]
                    h.correct[-1][test], h.cost[-1][test], h.latency[-1][test] = o.correct[test], o.cost[test], o.latency[test]
    return out


@dataclass(frozen=True)
class CapSummary:
    complete: bool  # every fold had a candidate within the cap
    accuracy: Accuracy | None  # mean of 4: per-item held-out score averaged over repeats, bootstrap CI over items
    per_question: dict[str, float]
    cost: float | None  # $ per 1,000 items, averaged over repeats
    latency: float | None  # p50 s, averaged over repeats


def summarise_held_out(h: HeldOut) -> CapSummary:
    if any(p.candidate is None for p in h.picks):
        return CapSummary(False, None, {}, None, None)
    correct = np.mean(h.correct, axis=0)  # (n, 4): averaged over repeats
    return CapSummary(
        True,
        accuracy(correct.mean(axis=1).tolist(), MEAN),
        {q: float(correct[:, j].mean()) for j, q in enumerate(QUESTIONS)},
        float(np.mean([c.mean() for c in h.cost]) * PER_THOUSAND),
        float(np.mean([np.median(lat) for lat in h.latency])),
    )
