"""The one-sentence "Key insight" under each chart on a run page, from the numbers that chart shows.

Pure: chart data in (asimov.plot_data, asimov.collect.tables), one sentence out. Facts and comparisons with
numbers only, never why: interpretation is the owners' (CLAUDE.md). Percentages are compared as the charts print
them (whole percents), so a sentence never disagrees with its chart. All wording is in TEMPLATES and the word
tables below.
"""

from __future__ import annotations

from collections.abc import Sequence

from asimov.bench.text_score import QUESTIONS
from asimov.claims import Claims
from asimov.collect.load import RunData
from asimov.collect.tables import SpeedRow
from asimov.plot_data import (
    Bar,
    CostPoint,
    GridRow,
    RangeCount,
    ScoringRow,
    TwinRow,
    accuracy_grid,
    cost_points,
    paper_in_range,
    severity_scores,
    text_vs_video,
    thinking_order,
    thinking_value,
    twin_rows,
)

# ---- wording: change it here ------------------------------------------------------------------
QUESTION_WORDS = {
    QUESTIONS[0]: "injury type",
    QUESTIONS[1]: "severity",
    QUESTIONS[2]: "action effect",
    QUESTIONS[3]: "severity after action",
}
METRIC_WORDS = {
    **{f"{q}_accuracy": w for q, w in QUESTION_WORDS.items()},
    "within_0_5_rate": "stepping in within 0.5 s",
    "timing_mae_s": "timing error",
}
# Comparison-version charts with paper values on their axes: figure -> (y metric, x metric), y first
RANGE_AXES = {
    "fig4a": (f"{QUESTIONS[0]}_accuracy", f"{QUESTIONS[1]}_accuracy"),
    "fig4b": (f"{QUESTIONS[2]}_accuracy", f"{QUESTIONS[3]}_accuracy"),
    "fig5b": ("within_0_5_rate", "timing_mae_s"),
}
TEMPLATES = {
    # comparison versions of fig4a, fig4b, fig5b
    "range_first": "on {axis}, the paper's value is inside our 95% range for {count}",
    "range_next": "on {axis}, for {count}",
    "range_missing": "the paper reports no {axis} values for these models",
    "range_none": "The paper reports no values for these models on this chart.",
    # fig5a
    "video_lead": "{who} {verb} {direction} on video than on text, by {amount} points",
    "video_lead_same": "{who} {verb} the same on video as on text",
    "video_next": "{k} {verb} {direction} (by {amount} points)",
    "video_next_same": "{k} {verb} the same",
    "video_none": "No model has both a text and a video score.",
    # accuracy heatmap
    "lowest": "{questions} {verb} the lowest-scoring question for {count}",
    "gap": "the largest gap to the paper is {cells}",
    "gaps": "the largest gaps to the paper ({points} points each) are {cells}",
    "gap_cell": "{name} on {question} ({ours}% vs {paper}%)",
    "gap_zero": "every value the paper reports matches ours to the whole percent",
    "gap_none": "the paper reports no values for these models",
    "grid_none": "No model has accuracy values.",
    # severity scoring
    "severity_lead": "Severe recall is {direction} than multiple-choice accuracy for {who}, by {amount} points",
    "severity_lead_same": "Severe recall equals multiple-choice accuracy for {who}",
    "severity_next": "it is {direction} for {k} (by {amount} points)",
    "severity_next_same": "it is equal for {k}",
    "severity_none": "No model has both a multiple-choice and a severe-recall score.",
    # accuracy vs cost
    "cost": "{names} {verb} the fewest output tokens per item ({fewest}, vs {most} for {most_names}, the most) and {be} {gap}.",
    "cost_best": "the most accurate model on every question",
    "cost_gap": "within {points} points of the most accurate model on every question",
    "cost_one": "{name} is the only model with token counts ({tokens} output tokens per item).",
    "cost_none": "No model reports output tokens.",
    # thinking twins
    "thinking": "From {low} to {high} thinking, {who} the most",
    "thinking_change": "{sign} points on {question}",
    "thinking_ratio": ", at {ratio:.1f}x the output tokens",
    "thinking_rest": "{others} by at most {points} points on any question",
    "thinking_one": "From {low} to {high} thinking, {name}'s largest change is {change}.",
    "thinking_zero": "From {low} to {high} thinking, no model's accuracy changes by a whole point on any question.",
    "thinking_none": "No model has accuracy in both runs.",
    # cost and speed table
    "speed": "{fast} {be} the fastest per reply (median {fast_s} s) and {slow} the slowest ({slow_s} s)",
    "speed_one": "{name} takes a median {seconds} s per reply",
    "tokens": "{most} {verb} the most output tokens per item ({most_n}) and {fewest} the fewest ({fewest_n})",
    "tokens_one": "writes {tokens} output tokens per item",
    "cost_all": "the whole run cost ${total:.2f}",
    "cost_some": "the {k} priced models cost ${total:.2f} in all",
    "speed_none": "No latency, token or price data for this run.",
}
LATENCY_DIGITS = 1  # as the run page prints latency


# ---- shared helpers -----------------------------------------------------------------------------


def pct(x: float) -> int:
    """A fraction as the charts print it: a whole percent."""
    return round(x * 100)


def models(k: int) -> str:
    return f"model{'' if k == 1 else 's'}"


def of(k: int, n: int) -> str:
    return f"{k} of {n} {models(n)}"


def joined(names: Sequence[str]) -> str:
    """ "A", "A and B", "A, B and C"."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def span(values: Sequence[int]) -> str:
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo} to {hi}"


def amount(values: Sequence[int], everyone: bool) -> str:
    """How much a group differs: the full range for every model, else the largest difference."""
    if everyone or len(values) == 1:
        return span(values)
    return f"up to {max(values)}"


def sentence(clauses: Sequence[str]) -> str:
    text = "; ".join(clauses)
    return text[0].upper() + text[1:] + "."


def by_direction(diffs: Sequence[tuple[str, int]], first: str) -> list[tuple[str, list[str], list[int]]]:
    """(direction, names, sizes) for the non-empty groups: `first` ("lower" or "higher"), the other, then "same"."""
    groups = {
        "lower": [(n, -d) for n, d in diffs if d < 0],
        "higher": [(n, d) for n, d in diffs if d > 0],
        "same": [(n, 0) for n, d in diffs if d == 0],
    }
    order = [first, "higher" if first == "lower" else "lower", "same"]
    return [(k, [n for n, _ in groups[k]], [d for _, d in groups[k]]) for k in order if groups[k]]


# ---- comparison versions of fig4a, fig4b, fig5b ---------------------------------------------------


def paper_range_insight(counts: Sequence[RangeCount]) -> str:
    """How many models' paper value sits inside our 95% CI, per axis."""
    if not any(c.with_paper for c in counts):
        return TEMPLATES["range_none"]
    clauses, first = [], True
    for c in counts:
        axis = METRIC_WORDS[c.metric]
        if not c.with_paper:
            clauses.append(TEMPLATES["range_missing"].format(axis=axis))
            continue
        clauses.append(TEMPLATES["range_first" if first else "range_next"].format(axis=axis, count=of(c.inside, c.with_paper)))
        first = False
    return sentence(clauses)


# ---- fig5a --------------------------------------------------------------------------------------


def text_vs_video_insight(bars: Sequence[Bar]) -> str:
    """Video minus text accuracy per model, grouped by direction."""
    diffs = [(b.name, pct(b.video) - pct(b.text)) for b in bars if b.text is not None and b.video is not None]
    if not diffs:
        return TEMPLATES["video_none"]
    n, clauses = len(diffs), []
    for i, (direction, names, sizes) in enumerate(by_direction(diffs, "lower")):
        k, everyone = len(names), len(names) == n
        verb = "scores" if k == 1 or everyone else "score"
        if i == 0:
            who = names[0] if n == 1 else "every model" if everyone else of(k, n)
            key = "video_lead_same" if direction == "same" else "video_lead"
            clauses.append(TEMPLATES[key].format(who=who, verb=verb, direction=direction, amount=amount(sizes, everyone)))
        else:
            key = "video_next_same" if direction == "same" else "video_next"
            clauses.append(TEMPLATES[key].format(k=k, verb=verb, direction=direction, amount=amount(sizes, False)))
    return sentence(clauses)


# ---- accuracy heatmap -----------------------------------------------------------------------------


def heatmap_insight(rows: Sequence[GridRow]) -> str:
    """The question most often lowest per model (ties count for each), and the largest gap to the paper."""
    counts, n = [0] * len(QUESTIONS), 0
    for r in rows:
        shown = [pct(v) for v in r.ours if v is not None]
        if not shown:
            continue
        n += 1
        for j, v in enumerate(r.ours):
            counts[j] += v is not None and pct(v) == min(shown)
    if not n:
        return TEMPLATES["grid_none"]
    top = max(counts)
    lowest = [f"Q{j + 1} ({QUESTION_WORDS[q]})" for j, q in enumerate(QUESTIONS) if counts[j] == top]
    lead = TEMPLATES["lowest"].format(questions=joined(lowest), verb="is" if len(lowest) == 1 else "are each", count=of(top, n))
    return sentence([lead, gap_clause(rows)])


def gap_clause(rows: Sequence[GridRow]) -> str:
    cells = [
        (abs(pct(o) - pct(p)), r.name, q, pct(o), pct(p))
        for r in rows
        for q, o, p in zip(QUESTIONS, r.ours, r.paper, strict=True)
        if o is not None and p is not None
    ]
    if not cells:
        return TEMPLATES["gap_none"]
    widest = max(c[0] for c in cells)
    if widest == 0:
        return TEMPLATES["gap_zero"]
    shown = [
        TEMPLATES["gap_cell"].format(name=name, question=QUESTION_WORDS[q], ours=o, paper=p)
        for g, name, q, o, p in cells
        if g == widest
    ]
    return (
        TEMPLATES["gap"].format(cells=shown[0])
        if len(shown) == 1
        else TEMPLATES["gaps"].format(points=widest, cells=joined(shown))
    )


# ---- severity scoring -----------------------------------------------------------------------------


def severity_insight(rows: Sequence[ScoringRow]) -> str:
    """Severe recall minus multiple-choice accuracy per model, grouped by direction."""
    diffs = [
        (r.name, pct(recall.value) - pct(mc.value))
        for r in rows
        if (mc := r.scores[0]) is not None and (recall := r.scores[-1]) is not None
    ]
    if not diffs:
        return TEMPLATES["severity_none"]
    n, clauses = len(diffs), []
    for i, (direction, names, sizes) in enumerate(by_direction(diffs, "higher")):
        everyone = len(names) == n
        if i == 0:
            who = names[0] if n == 1 else "every model" if everyone else of(len(names), n)
            key = "severity_lead_same" if direction == "same" else "severity_lead"
            clauses.append(TEMPLATES[key].format(who=who, direction=direction, amount=amount(sizes, everyone)))
        else:
            key = "severity_next_same" if direction == "same" else "severity_next"
            clauses.append(TEMPLATES[key].format(k=len(names), direction=direction, amount=amount(sizes, False)))
    return sentence(clauses)


# ---- accuracy vs cost -----------------------------------------------------------------------------


def cost_insight(points: Sequence[CostPoint]) -> str:
    """The model(s) with the fewest output tokens per item, and their largest gap to the best model on any question."""
    shown = [(p, round(p.tokens)) for p in points if p.tokens is not None]
    if not shown:
        return TEMPLATES["cost_none"]
    if len(shown) == 1:
        return TEMPLATES["cost_one"].format(name=shown[0][0].name, tokens=f"{shown[0][1]:,}")
    fewest, most = min(t for _, t in shown), max(t for _, t in shown)
    cheap = [p for p, t in shown if t == fewest]
    best = [max((pct(a) for p, _ in shown if (a := p.accuracy[j]) is not None), default=None) for j in range(len(QUESTIONS))]
    gap = max(
        (b - pct(a) for p in cheap for a, b in zip(p.accuracy, best, strict=True) if a is not None and b is not None), default=0
    )
    return TEMPLATES["cost"].format(
        names=joined([p.name for p in cheap]),
        verb="uses" if len(cheap) == 1 else "use",
        fewest=f"{fewest:,}",
        most=f"{most:,}",
        most_names=joined([p.name for p, t in shown if t == most]),
        be="is" if len(cheap) == 1 else "are",
        gap=TEMPLATES["cost_best"] if gap == 0 else TEMPLATES["cost_gap"].format(points=gap),
    )


# ---- thinking twins -------------------------------------------------------------------------------


def biggest_change(r: TwinRow) -> tuple[int, str] | None:
    """(high minus low points, question) with the largest size, the first question on a tie; None without data."""
    changes = [
        (pct(h) - pct(lo), q) for q, lo, h in zip(QUESTIONS, r.low, r.high, strict=True) if lo is not None and h is not None
    ]
    return max(changes, key=lambda c: abs(c[0])) if changes else None


def change_text(r: TwinRow, change: tuple[int, str]) -> str:
    text = TEMPLATES["thinking_change"].format(sign=f"{change[0]:+d}", question=QUESTION_WORDS[change[1]])
    ratio = r.token_ratio
    return text + ("" if ratio is None else TEMPLATES["thinking_ratio"].format(ratio=ratio))


def thinking_insight(rows: Sequence[TwinRow], low: str, high: str) -> str:
    """The model whose accuracy changes most between the two thinking values (with its token ratio), and the rest's largest."""
    found = [(r, c) for r in rows if (c := biggest_change(r)) is not None]
    if not found:
        return TEMPLATES["thinking_none"]
    widest = max(abs(c[0]) for _, c in found)
    if widest == 0:
        return TEMPLATES["thinking_zero"].format(low=low, high=high)
    if len(found) == 1:
        r, c = found[0]
        return TEMPLATES["thinking_one"].format(low=low, high=high, name=r.name, change=change_text(r, c))
    top = [(r, c) for r, c in found if abs(c[0]) == widest]
    if len(top) == 1:
        who = f"{top[0][0].name} changes"
        lead = TEMPLATES["thinking"].format(low=low, high=high, who=who) + f" ({change_text(*top[0])})"
    else:
        who = joined([f"{r.name} ({change_text(r, c)})" for r, c in top]) + " change"
        lead = TEMPLATES["thinking"].format(low=low, high=high, who=who)
    rest = [abs(c[0]) for r, c in found if abs(c[0]) != widest]
    if not rest:
        return sentence([lead])
    others = "the other model changes" if len(rest) == 1 else "every other model changes"
    return sentence([lead, TEMPLATES["thinking_rest"].format(others=others, points=max(rest))])


# ---- cost and speed table -------------------------------------------------------------------------


def extremes(values: Sequence[tuple[str, float]]) -> tuple[list[str], float, list[str], float]:
    lo, hi = min(v for _, v in values), max(v for _, v in values)
    return [n for n, v in values if v == lo], lo, [n for n, v in values if v == hi], hi


def speed_insight(rows: Sequence[SpeedRow]) -> str:
    """Fastest and slowest median latency, most and fewest output tokens per item, and the run's cost when priced."""
    latency = [(r.name, round(r.latency_p50_s, LATENCY_DIGITS)) for r in rows if r.latency_p50_s is not None]
    tokens = [(r.name, round(r.output_per_item)) for r in rows if r.output_per_item is not None]
    costs = [r.cost for r in rows if r.cost is not None]
    if not (latency or tokens or costs):
        return TEMPLATES["speed_none"]
    if len(rows) == 1:
        r = rows[0]
        parts = [TEMPLATES["speed_one"].format(name=r.name, seconds=latency[0][1])] if latency else [r.name]
        parts += [TEMPLATES["tokens_one"].format(tokens=f"{tokens[0][1]:,}")] if tokens else []
        return sentence([" and ".join(parts)] + cost_clause(costs, len(rows)))
    clauses = []
    if latency:
        fast, fast_s, slow, slow_s = extremes(latency)
        be = "is" if len(fast) == 1 else "are"
        clauses.append(TEMPLATES["speed"].format(fast=joined(fast), be=be, fast_s=fast_s, slow=joined(slow), slow_s=slow_s))
    if tokens:
        few, few_n, most, most_n = extremes(tokens)
        verb = "writes" if len(most) == 1 else "write"
        clauses.append(
            TEMPLATES["tokens"].format(
                most=joined(most), verb=verb, most_n=f"{most_n:,}", fewest=joined(few), fewest_n=f"{few_n:,}"
            )
        )
    return sentence(clauses + cost_clause(costs, len(rows)))


def cost_clause(costs: Sequence[float], n: int) -> list[str]:
    if not costs:
        return []
    if len(costs) == n:
        return [TEMPLATES["cost_all"].format(total=sum(costs))]
    return [TEMPLATES["cost_some"].format(k=len(costs), total=sum(costs))]


# ---- per chart on a run page ----------------------------------------------------------------------


def comparison_insight(fig: str, run: RunData, partner: RunData | None, claims: Claims) -> str | None:
    """The sentence under a comparison-version figure (fig4a, fig4b, fig5a, fig5b); None for one without."""
    if fig in RANGE_AXES:
        return paper_range_insight([paper_in_range(run, claims, fig[3:], k) for k in RANGE_AXES[fig]])
    if fig == "fig5a" and partner is not None:
        text, video = (run, partner) if run.benchmark == "text" else (partner, run)
        return text_vs_video_insight(text_vs_video(text, video, claims, only_paper_models=False))
    return None


def extra_insight(kind: str, run: RunData, claims: Claims, twin: RunData | None = None) -> str | None:
    """The sentence under one of our own charts (pages.EXTRA_FIGURES); `twin` for a thinking comparison."""
    if kind == "accuracy_heatmap":
        return heatmap_insight(accuracy_grid(run, claims))
    if kind == "severity_scoring":
        return severity_insight(severity_scores(run, claims))
    if kind == "accuracy_vs_cost":
        return cost_insight(cost_points(run))
    if kind == "thinking_vs" and twin is not None:
        low, high = thinking_order(run, twin)
        return thinking_insight(twin_rows(low, high), thinking_value(low), thinking_value(high))
    return None
