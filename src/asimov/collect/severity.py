"""Ways to score the text benchmark's severity questions, side by side, from fields already in summary.json.

Fig 4 uses the exact-level score (`<q>_accuracy`): the paper's Fig 3 defines severity that way
(Noa, WORKLOG 2026-10-12). The other scorings come from `bench.text_score.severity_metrics`;
nothing here re-scores a reply. A field missing from an older summary stays None.
"""

from __future__ import annotations

from dataclasses import dataclass

from asimov.claims import Claims, paper_value
from asimov.collect.load import ModelResult, RunData

# Plain-language definition of the Fig 4 severity score; pages and figures say it this way.
EXACT_LEVEL = (
    "the share of multiple-choice severity questions answered with the exact right level (none / minor / moderate / severe)"
)
# scoring -> summary key suffix, for the paper-gap line (all compared to the paper's Fig 4a values)
SCORINGS = {
    "exact level right": "_accuracy",
    "severe vs not right": "_severe_binary_accuracy",
    "truly severe caught": "_severe_recall",
}
PAPER_FIGURE = {"latent_severity": "4a", "activated_risk": "4b"}  # where the paper plots each question


@dataclass(frozen=True)
class SeverityRow:
    model: str
    exact: float | None
    severe_vs_not: float | None
    caught: float | None
    n_severe: int | None
    false_negatives: int | None
    false_negative_rate: float | None
    false_positives: int | None
    false_positive_rate: float | None
    n_not_severe: int | None


def severity_row(m: ModelResult, question: str) -> SeverityRow:
    s = m.summary
    n_severe = s.get(f"{question}_n_severe")
    n_not_severe = None if n_severe is None or s.get("n") is None else s["n"] - n_severe
    return SeverityRow(
        m.name,
        s.get(f"{question}_accuracy"),
        s.get(f"{question}_severe_binary_accuracy"),
        s.get(f"{question}_severe_recall"),
        n_severe,
        s.get(f"{question}_severe_false_negatives"),
        s.get(f"{question}_severe_false_negative_rate"),
        s.get(f"{question}_severe_false_positives"),
        s.get(f"{question}_severe_false_positive_rate"),
        n_not_severe,
    )


def mean_abs_gap(pairs: list[tuple[float | None, float | None]]) -> float | None:
    """Mean |ours - paper| over the pairs where both values exist; None if there are none."""
    gaps = [abs(a - b) for a, b in pairs if a is not None and b is not None]
    return sum(gaps) / len(gaps) if gaps else None


def paper_gaps(run: RunData, claims: Claims, question: str) -> tuple[dict[str, float | None], int]:
    """Each scoring's mean gap to the paper's plotted severity values for `question`, and the models compared."""
    metric, figure = f"{question}_accuracy", PAPER_FIGURE[question]
    paper = {m.name: paper_value(claims, run.benchmark, m.name, metric, figure) for m in run.models}
    gaps = {
        scoring: mean_abs_gap([(m.summary.get(question + suffix), paper[m.name]) for m in run.models])
        for scoring, suffix in SCORINGS.items()
    }
    return gaps, sum(p is not None for p in paper.values())
