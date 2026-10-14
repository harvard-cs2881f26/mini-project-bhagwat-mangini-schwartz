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
6. split_by_question (design B): the decision model answers some questions on every item, the LLM the rest. Both
   calls are paid, one after the other (latency: the sum); the 2 one-sided splits are the single systems.
7. parallel: the decision model and the LLM are called at once on every item (both paid; latency: the slower).
   The LLM's answers are kept on the lowest-confidence share of items, the decision model's elsewhere.
8. parallel_per_question_mix: parallel, with per_question_mix's per-question choice on the low-confidence items.

Also: the perfect-router bound per decision model x LLM (a bound, not a system; asimov.cascade.gain_order) and
Q2-Q4 accuracy on the items whose Q1 matched the key (WORKLOG 2026-10-14, "Cumulative-mode scoring").
GPU seconds: a self-hosted member's latency counts as GPU time on each call it makes, added up even when calls run
in parallel. WORKLOG 2026-10-14, "Router search: rerun + extensions (exploratory)".
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from asimov.bench.text_score import QUESTIONS
from asimov.cascade import MEAN, PER_THOUSAND, Accuracy, accuracy, gain_order, system_from
from asimov.collect.load import ModelResult
from asimov.prices import Price

Rule = Literal["any", "q1"]
CapKind = Literal["cost", "latency"]
FAMILIES = (
    "single",
    "d_cascade",
    "agreement",
    "three_stage",
    "per_question_mix",
    "split_by_question",
    "parallel",
    "parallel_per_question_mix",
)
Q1 = QUESTIONS[0]  # latent_risk: the question the second scoring rule conditions on
UNANSWERED = ""  # a missing answer never agrees with anything


class PoolEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str  # run name (config `run`)
    model: str  # model id in that run
    label: str  # unique name in the tables
    note: str | None = None  # printed beside the pools in tables.md (e.g. a known problem with the run)


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
    compute_rows: int = 0  # items whose latency is Modal compute time (raw.remote.compute_s), not the round trip
    params_b: float | None = None  # parameters in billions (configs/prices.yaml), where stated

    @property
    def gpu(self) -> np.ndarray:
        """GPU seconds per item: a self-hosted model's latency (compute time where saved, else the round trip)."""
        return self.latency if self.self_hosted else np.zeros_like(self.latency)


def member_from(label: str, model: ModelResult, price: Price, ids: Sequence[str], rule: str | None) -> Member:
    """Validated by cascade.system_from (no failed calls, usage and latency for every item).

    Latency: a reply made on Modal counts its GPU compute time (raw.remote.compute_s), because its round-trip latency_s
    includes queue wait when the run used several workers; every other reply counts latency_s."""
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
        np.array([model.item_compute_s.get(i, it.latency) for i, it in zip(ids, items, strict=True)], dtype=float),
        np.array([system.confidence[i] for i in ids]) if rule else None,
        bool(price.self_hosted),
        sum(i in model.item_compute_s for i in ids),
        price.params_b,
    )


@dataclass(frozen=True)
class Outcome:
    correct: np.ndarray  # (n, 4)
    cost: np.ndarray  # (n,)
    latency: np.ndarray  # (n,)
    gpu: np.ndarray  # (n,) GPU seconds of self-hosted calls, added up


@dataclass(frozen=True)
class Candidate:
    family: str
    name: str  # its full definition
    self_hosted: bool  # uses a self-hosted model, whose cost is not in $
    fit: Callable[[np.ndarray], Outcome]  # training item indices -> outcome on every item
    members: tuple[str, ...] = ()  # labels of the systems it uses (for the notes printed beside them)


def alone(m: Member) -> Outcome:
    return Outcome(m.correct, m.cost, m.latency, m.gpu)


def hosted(*ms: Member) -> bool:
    return any(m.self_hosted for m in ms)


def labels(*ms: Member) -> tuple[str, ...]:
    return tuple(m.label for m in ms)


def sent_below(confidence: np.ndarray, train: np.ndarray, fraction: float) -> np.ndarray:
    """Items whose confidence is below the `fraction` quantile of the training items' confidences."""
    return confidence < np.quantile(confidence[train], fraction)


def escalate(base: Outcome, to: Member, sent: np.ndarray, keep: np.ndarray | None = None) -> Outcome:
    """Sent items also call `to` (sequentially); its answers replace base's except on `keep` questions."""
    take = sent[:, None] & ~(np.zeros(len(QUESTIONS), bool) if keep is None else keep)[None, :]
    return Outcome(
        np.where(take, to.correct, base.correct),
        base.cost + to.cost * sent,
        base.latency + to.latency * sent,
        base.gpu + to.gpu * sent,
    )


def in_parallel(a: Member, b: Member) -> Outcome:
    """a's answers, with a and b both called at once on every item: both paid, the slower one's latency."""
    return Outcome(a.correct, a.cost + b.cost, np.maximum(a.latency, b.latency), a.gpu + b.gpu)


def take_answers(base: Outcome, src: Member, sent: np.ndarray, keep: np.ndarray | None = None) -> Outcome:
    """src was already called on every item (in parallel): its answers replace base's on sent items, except on `keep`
    questions. Cost and latency don't change."""
    take = sent[:, None] & ~(np.zeros(len(QUESTIONS), bool) if keep is None else keep)[None, :]
    return Outcome(np.where(take, src.correct, base.correct), base.cost, base.latency, base.gpu)


def better_on_training(dec: Member, llm: Member, train: np.ndarray, sent: np.ndarray) -> np.ndarray | None:
    """Per question, True where the decision model beat the LLM on the training items sent on (ties: the LLM)."""
    mine = train[sent[train]]
    return dec.correct[mine].mean(axis=0) > llm.correct[mine].mean(axis=0) if len(mine) else None


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
    return Candidate("single", m.label, m.self_hosted, lambda train: alone(m), labels(m))


def d_cascade(dec: Member, llm: Member, fraction: float) -> Candidate:
    conf = needs_confidence(dec)
    name = f"{dec.label} -> {llm.label} (lowest {fraction:.0%} by confidence)"
    return Candidate(
        "d_cascade",
        name,
        hosted(dec, llm),
        lambda t: escalate(alone(dec), llm, sent_below(conf, t, fraction)),
        labels(dec, llm),
    )


def per_question_mix(dec: Member, llm: Member, fraction: float) -> Candidate:
    conf = needs_confidence(dec)

    def fit(train: np.ndarray) -> Outcome:
        sent = sent_below(conf, train, fraction)
        return escalate(alone(dec), llm, sent, better_on_training(dec, llm, train, sent))

    name = f"{dec.label} -> {llm.label} (lowest {fraction:.0%} by confidence; per question, the better on training)"
    return Candidate("per_question_mix", name, hosted(dec, llm), fit, labels(dec, llm))


def agreement(a: Member, b: Member, strong: Member, rule: Rule) -> Candidate:
    """a and b answer in parallel (latency: the slower); b's answers are kept where they agree under `rule`."""
    first = in_parallel(b, a)
    sent = disagree(a.pred, b.pred, rule)
    name = f"{a.label} + {b.label} agree ({rule}) else {strong.label}"
    return Candidate("agreement", name, hosted(a, b, strong), lambda train: escalate(first, strong, sent), labels(a, b, strong))


def three_stage(dec: Member, cheap: Member, strong: Member, fraction: float, rule: Rule) -> Candidate:
    conf = needs_confidence(dec)
    differ = disagree(dec.pred, cheap.pred, rule)

    def fit(train: np.ndarray) -> Outcome:
        low = sent_below(conf, train, fraction)
        return escalate(escalate(alone(dec), cheap, low), strong, low & differ)

    name = f"{dec.label} -> {cheap.label} (lowest {fraction:.0%}) -> {strong.label} if they disagree ({rule})"
    return Candidate("three_stage", name, hosted(dec, cheap, strong), fit, labels(dec, cheap, strong))


def split_by_question(dec: Member, llm: Member, from_decision: tuple[bool, ...]) -> Candidate:
    """Design B: each question's answer from the decision model (True) or the LLM (False), on every item. A side
    that answers nothing isn't called; otherwise both calls are paid, one after the other."""
    keep = np.array(from_decision, dtype=bool)
    calls = [m for m, used in ((dec, keep.any()), (llm, not keep.all())) if used]
    correct = np.where(keep[None, :], dec.correct, llm.correct)
    outcome = Outcome(
        correct,
        np.sum([m.cost for m in calls], axis=0),
        np.sum([m.latency for m in calls], axis=0),  # sequential calls
        np.sum([m.gpu for m in calls], axis=0),
    )
    mine = ", ".join(f"Q{k + 1}" for k in np.flatnonzero(keep)) or "none"
    name = f"{dec.label} answers {mine}; {llm.label} the rest"
    return Candidate("split_by_question", name, hosted(*calls), lambda train: outcome, labels(dec, llm))


def parallel(dec: Member, llm: Member, fraction: float) -> Candidate:
    conf = needs_confidence(dec)
    both = in_parallel(dec, llm)
    name = f"{dec.label} || {llm.label} in parallel; {llm.label}'s answers on the lowest {fraction:.0%} by confidence"
    return Candidate(
        "parallel", name, hosted(dec, llm), lambda t: take_answers(both, llm, sent_below(conf, t, fraction)), labels(dec, llm)
    )


def parallel_per_question_mix(dec: Member, llm: Member, fraction: float) -> Candidate:
    conf = needs_confidence(dec)
    both = in_parallel(dec, llm)

    def fit(train: np.ndarray) -> Outcome:
        sent = sent_below(conf, train, fraction)
        return take_answers(both, llm, sent, better_on_training(dec, llm, train, sent))

    name = (
        f"{dec.label} || {llm.label} in parallel; {llm.label}'s answers on the lowest {fraction:.0%} by confidence "
        "(per question, the better on training)"
    )
    return Candidate("parallel_per_question_mix", name, hosted(dec, llm), fit, labels(dec, llm))


SPLITS = [mask for mask in itertools.product((True, False), repeat=len(QUESTIONS)) if any(mask) and not all(mask)]


def candidates(
    decisions: Sequence[Member],
    llms: Sequence[Member],
    cheap: Sequence[Member],
    strong: Sequence[Member],
    fractions: Sequence[float],
    rules: Sequence[Rule],
) -> list[Candidate]:
    """The finite search space. Agreement pairs: decision + decision (unordered for `any`, where the kept answers are
    the same; ordered for `q1`), and decision + cheap LLM (the LLM's answers kept). Split by question: the 14 splits
    where both sides answer something (the other 2 are the single systems)."""
    out = [single(m) for m in [*decisions, *llms]]
    out += [d_cascade(d, m, f) for d in decisions for m in llms for f in fractions]
    for rule in rules:
        pairs = [(a, b) for i, a in enumerate(decisions) for j, b in enumerate(decisions) if (i < j if rule == "any" else i != j)]
        pairs += [(d, c) for d in decisions for c in cheap]
        out += [agreement(a, b, s, rule) for a, b in pairs for s in strong]
    out += [three_stage(d, c, s, f, r) for r in rules for d in decisions for c in cheap for s in strong for f in fractions]
    out += [per_question_mix(d, m, f) for d in decisions for m in llms for f in fractions]
    out += [split_by_question(d, m, mask) for d in decisions for m in llms for mask in SPLITS]
    out += [parallel(d, m, f) for d in decisions for m in llms for f in fractions]
    out += [parallel_per_question_mix(d, m, f) for d in decisions for m in llms for f in fractions]
    return out


# --- perfect-router bound and the Q1-matched rule --------------------------------------------------------------


def perfect_router_bound(dec: Member, llm: Member) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The best possible per-item routing (a bound, not a system): items go to the LLM, after the decision model, only
    where the LLM gets strictly more of the 4 questions right, in asimov.cascade.gain_order. Fitted on every item by
    definition. Returns items sent, mean of 4 and $ per 1,000 items, one entry per number of items sent."""
    dec_score, llm_score = dec.correct.mean(axis=1), llm.correct.mean(axis=1)
    gain = llm_score - dec_score
    order = np.array(gain_order(dict(enumerate(gain.tolist())), dict(enumerate(llm.cost.tolist()))), dtype=int)
    n = len(gain)
    acc = (dec_score.sum() + np.concatenate([[0.0], np.cumsum(gain[order])])) / n
    cost = (dec.cost.sum() + np.concatenate([[0.0], np.cumsum(llm.cost[order])])) / n * PER_THOUSAND
    return np.arange(len(order) + 1), acc, cost


def q1_matched(correct: np.ndarray) -> tuple[float, dict[str, float | None]]:
    """Second scoring rule (Noa's option c): Q2-Q4 accuracy on the items whose Q1 answer matched the key.
    `correct`: (repeats, n, 4). Matched items are pooled over repeats; also returns the mean number matched per repeat.
    None for a question when no item matched."""
    matched = correct[..., 0] == 1
    total = int(matched.sum())
    acc = {q: (float(correct[..., j][matched].mean()) if total else None) for j, q in enumerate(QUESTIONS) if j}
    return total / correct.shape[0], acc


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
    gpu: list[np.ndarray] = field(default_factory=list)  # GPU seconds of self-hosted calls

    def start_repeat(self, n: int) -> None:
        self.correct.append(np.full((n, len(QUESTIONS)), np.nan))
        for part in (self.cost, self.latency, self.gpu):
            part.append(np.full(n, np.nan))

    def record(self, o: Outcome, test: np.ndarray) -> None:
        self.correct[-1][test] = o.correct[test]
        self.cost[-1][test], self.latency[-1][test], self.gpu[-1][test] = o.cost[test], o.latency[test], o.gpu[test]


def within(m: Metrics, idx: np.ndarray) -> Metrics:
    return Metrics(m.accuracy[idx], m.cost[idx], m.latency[idx])


def cross_validate_groups(
    cands: Sequence[Candidate],
    n: int,
    folds: int,
    seeds: Sequence[int],
    caps: Mapping[str, Sequence[float]],
    groups: Mapping[str, np.ndarray],
) -> dict[tuple[str, str, float], HeldOut]:
    """For each repeat and fold: fit every candidate on the training folds, pick per group (candidate indices) and cap
    on them, and record the pick's outcome on the held-out fold. Picks are indices into `cands`."""
    out = {(g, kind, cap): HeldOut() for g in groups for kind, values in caps.items() for cap in values}
    for seed in seeds:
        fold = fold_ids(n, folds, seed)
        for h in out.values():
            h.start_repeat(n)
        for k in range(folds):
            train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            outcomes = [c.fit(train) for c in cands]
            on_train = metrics_on(outcomes, train)
            for (g, kind, cap), h in out.items():
                local = select(within(on_train, groups[g]), kind, cap)  # pyright: ignore[reportArgumentType]
                pick = None if local is None else int(groups[g][local])
                h.picks.append(Pick(seed, k, pick))
                if pick is not None:
                    h.record(outcomes[pick], test)
    return out


def cross_validate(
    cands: Sequence[Candidate], n: int, folds: int, seeds: Sequence[int], caps: Mapping[str, Sequence[float]]
) -> dict[tuple[str, float], HeldOut]:
    """cross_validate_groups with one group: every candidate."""
    cv = cross_validate_groups(cands, n, folds, seeds, caps, {"all": np.arange(len(cands))})
    return {(kind, cap): h for (_, kind, cap), h in cv.items()}


@dataclass(frozen=True)
class CapSummary:
    complete: bool  # every fold had a candidate within the cap
    accuracy: Accuracy | None  # mean of 4: per-item held-out score averaged over repeats, bootstrap CI over items
    per_question: dict[str, float]
    cost: float | None  # $ per 1,000 items, averaged over repeats
    latency: float | None  # p50 s, averaged over repeats
    gpu: float | None = None  # mean GPU seconds per item (self-hosted calls), averaged over repeats
    n_q1_matched: float | None = None  # items whose Q1 matched the key, mean per repeat
    q1_matched: dict[str, float | None] = field(default_factory=dict)  # Q2-Q4 accuracy on those items


def summarise_held_out(h: HeldOut) -> CapSummary:
    if any(p.candidate is None for p in h.picks):
        return CapSummary(False, None, {}, None, None)
    correct = np.mean(h.correct, axis=0)  # (n, 4): averaged over repeats
    n_matched, matched = q1_matched(np.stack(h.correct))
    return CapSummary(
        True,
        accuracy(correct.mean(axis=1).tolist(), MEAN),
        {q: float(correct[:, j].mean()) for j, q in enumerate(QUESTIONS)},
        float(np.mean([c.mean() for c in h.cost]) * PER_THOUSAND),
        float(np.mean([np.median(lat) for lat in h.latency])),
        float(np.mean([g.mean() for g in h.gpu])),
        n_matched,
        matched,
    )
