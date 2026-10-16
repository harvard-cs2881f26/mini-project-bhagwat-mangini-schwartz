"""Which metrics each benchmark reports, their 95% CIs, and per-item correctness by question.

Metric values always come from the scorer's summary.json; nothing here re-scores a reply. A CI
needs the metric's denominator: Wilson on round(value x n) of n, where n is the summary key
named in DENOMINATORS. A mean metric (timing error) gets a bootstrap CI over its per-item values.
"""

from __future__ import annotations

from dataclasses import dataclass

from asimov.bench.text_score import QUESTIONS, SEVERITY_QUESTIONS
from asimov.collect.load import ModelResult
from asimov.stats import bootstrap_mean_ci, holm, mcnemar_exact, paired_bootstrap_diff, wilson_ci

# benchmark -> reported metric -> summary key holding its denominator, or None for a per-item mean.
# Image is provisional until its scorer exists (Riddhi).
DENOMINATORS: dict[str, dict[str, str | None]] = {
    "text": {
        **{f"{q}_accuracy": "n" for q in QUESTIONS},
        **{f"{q}_severe_binary_accuracy": "n" for q in SEVERITY_QUESTIONS},
        **{f"{q}_severe_recall": f"{q}_n_severe" for q in SEVERITY_QUESTIONS},
    },
    "video": {"risk_accuracy": "n", "within_0_5_rate": "n", "timing_mae_s": None, "severity_accuracy": "n_severity"},
    "image": {"violation_rate": "n"},
}
MEAN_SOURCES = {"timing_mae_s": "error_s"}  # per-item field averaged by the scorer (video: "error" cases only)
VIDEO_QUESTIONS = ("risk", "timing")  # per_item.csv rows per video item: risk = risk_ok, timing = within_0_5 (WORKLOG 2026-10-12)


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float | None
    hi: float | None
    n: int  # items behind the CI

    def contains(self, x: float) -> bool | None:
        return None if self.lo is None or self.hi is None else self.lo <= x <= self.hi


def estimate(model: ModelResult, benchmark: str, metric: str) -> Estimate | None:
    """The summary's value for a metric with its 95% CI; None if the summary doesn't have it."""
    value = model.summary.get(metric)
    if value is None:
        return None
    denominators = DENOMINATORS.get(benchmark, {})
    if metric not in denominators:
        return Estimate(value, None, None, model.summary.get("n", 0))
    den = denominators[metric]
    if den is None:
        field = MEAN_SOURCES[metric]
        values = [o[field] for o in model.per_item.values() if o.get(field) is not None]
        if not values:
            return Estimate(value, None, None, 0)
        _, lo, hi = bootstrap_mean_ci(values)
        return Estimate(value, lo, hi, len(values))
    n = model.summary.get(den) or 0
    if not n:
        return Estimate(value, None, None, 0)
    lo, hi = wilson_ci(round(value * n), n)
    return Estimate(value, lo, hi, n)


def item_rows(benchmark: str, outcome: dict) -> list[dict]:
    """One row per question for one item's outcome: question, pred, gold, correct, status."""
    if benchmark == "text":
        return [
            {
                "question": q,
                "pred": outcome[q]["pred"],
                "gold": outcome[q]["gold"],
                "correct": outcome[q]["correct"],
                "status": outcome.get("parse", {}).get(q, ""),
            }
            for q in QUESTIONS
        ]
    if benchmark == "video":
        risk, timing = VIDEO_QUESTIONS
        return [
            {"question": risk, "pred": None, "gold": None, "correct": outcome["risk_ok"], "status": ""},
            {
                "question": timing,
                "pred": outcome["pred_s"],
                "gold": outcome["truth_s"],
                "correct": outcome["within_0_5"],
                "status": outcome["case"],
            },
        ]
    return [
        {"question": q, "pred": o.get("pred"), "gold": o.get("gold"), "correct": o["correct"], "status": ""}
        for q, o in outcome.items()
        if isinstance(o, dict) and "correct" in o
    ]


def correct_by_question(model: ModelResult, benchmark: str) -> dict[str, dict[str, bool]]:
    """question -> item id -> correct, for paired comparisons."""
    out: dict[str, dict[str, bool]] = {}
    for item_id, outcome in model.per_item.items():
        for row in item_rows(benchmark, outcome):
            out.setdefault(row["question"], {})[item_id] = bool(row["correct"])
    return out


@dataclass(frozen=True)
class Paired:
    n: int
    diff: float  # mean(a) - mean(b)
    lo: float
    hi: float
    a_only: int  # items a got right and b wrong
    b_only: int
    p: float  # exact McNemar
    a_acc: float  # mean(a) on the common items
    b_acc: float


def paired(a: dict[str, bool], b: dict[str, bool]) -> Paired | None:
    """a vs b on the items both scored (sorted ids, so the bootstrap is reproducible)."""
    common = sorted(a.keys() & b.keys())
    if not common:
        return None
    va, vb = [a[i] for i in common], [b[i] for i in common]
    diff, lo, hi = paired_bootstrap_diff(va, vb)
    a_only = sum(x and not y for x, y in zip(va, vb, strict=True))
    b_only = sum(y and not x for x, y in zip(va, vb, strict=True))
    return Paired(len(common), diff, lo, hi, a_only, b_only, mcnemar_exact(a_only, b_only), sum(va) / len(va), sum(vb) / len(vb))


@dataclass(frozen=True)
class PairedMean:
    n: int
    a_acc: float  # per item, mean correct over the questions; then averaged over items
    b_acc: float
    diff: float
    lo: float
    hi: float


def paired_mean(a: dict[str, dict[str, bool]], b: dict[str, dict[str, bool]]) -> PairedMean | None:
    """Mean over questions per item, a vs b, on the items both scored on every question (paired bootstrap CI, no
    test: there is no exact paired test for a 0-1 average). None for fewer than two questions in common."""
    questions = sorted(a.keys() & b.keys())
    if len(questions) < 2:
        return None
    common = sorted(set.intersection(*(set(a[q]) & set(b[q]) for q in questions)))
    if not common:
        return None
    va = [sum(a[q][i] for q in questions) / len(questions) for i in common]
    vb = [sum(b[q][i] for q in questions) / len(questions) for i in common]
    diff, lo, hi = paired_bootstrap_diff(va, vb)
    return PairedMean(len(common), sum(va) / len(va), sum(vb) / len(vb), diff, lo, hi)


def holm_by_family(
    raw: dict[str, dict[tuple[str, str], float]], family: dict[str, str | None]
) -> dict[str, dict[tuple[str, str], float]]:
    """comparison -> (model, question) -> raw p, to the Holm-adjusted p across every test of the comparisons that share
    a family (stats.holm). A comparison with no family gets an empty dict: raw p only."""
    out: dict[str, dict[tuple[str, str], float]] = {name: {} for name in raw}
    for fam in {f for f in family.values() if f}:
        keys = [(name, k) for name in raw if family.get(name) == fam for k in raw[name]]
        for (name, k), adj in zip(keys, holm([raw[name][k] for name, k in keys]), strict=True):
            out[name][k] = adj
    return out
