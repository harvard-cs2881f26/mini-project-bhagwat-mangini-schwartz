"""What each paper figure plots (docs/PLAN.md §10), as plain data: pure functions, no matplotlib.

Values and CIs come from asimov.collect.metrics.estimate, the same code the results pages use, so
a figure and its page can't disagree. Paper values come from configs/paper/claims.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

from asimov.claims import Claims, as_fraction, model_key
from asimov.collect.load import RunData
from asimov.collect.metrics import estimate

# Fig 4a/4b severity axis: Noa picks which goes in the report; 4-way is the default (WORKLOG 2026-10-12).
SEVERITY = {"4way": "_accuracy", "binary": "_severe_binary_accuracy", "recall": "_severe_recall"}
# Goes between the axis name and "(Higher is Better)", e.g. "Severity Accuracy, severe vs not (Higher is Better)".
SEVERITY_LABEL = {"4way": "", "binary": ", severe vs not", "recall": ", severe recall"}
# Validated categorical palette (dataviz skill, light mode on the grey plot surface): one hue per model,
# by the model's position in the Fig 4 config, so a model keeps its color in every figure.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
FALLBACK_COLOR = "#6b6b6b"
SAME_SPOT_DECIMALS = 6  # points equal to this many decimals are drawn as one spot (ringed markers)
MAX_LABEL_SHIFTS = 12  # slots tried above and below a label's home before it is left where it is


@dataclass(frozen=True)
class Point:
    name: str  # our config name, e.g. "Claude Opus 4.1 (4.5)"
    x: float | None
    y: float | None
    x_ci: tuple[float, float] | None = None
    y_ci: tuple[float, float] | None = None


def severity_metric(question: str, severity: str) -> str:
    return question + SEVERITY[severity]


def paper_value(claims: Claims, benchmark: str, name: str, metric: str, figure: str) -> float | None:
    """The paper's single-model value for this figure, on our scale."""
    for c in claims.claims:
        same_model = model_key(c.model or "") == model_key(name)
        if c.kind == "value" and c.figure == figure and c.benchmark == benchmark and c.metric == metric and same_model:
            return as_fraction(c)
    return None


def ours(run: RunData, x_metric: str, y_metric: str, y_scale: float = 1.0) -> list[Point]:
    points = []
    for m in run.models:
        ex, ey = estimate(m, run.benchmark, x_metric), estimate(m, run.benchmark, y_metric)
        if ex is None or ey is None:
            continue
        x_ci = (ex.lo, ex.hi) if ex.lo is not None and ex.hi is not None else None
        y_ci = (ey.lo * y_scale, ey.hi * y_scale) if ey.lo is not None and ey.hi is not None else None
        points.append(Point(m.name, ex.value, ey.value * y_scale, x_ci, y_ci))
    return points


def paper(run: RunData, claims: Claims, figure: str, x_metric: str, y_metric: str, y_scale: float = 1.0) -> list[Point]:
    """Paper points for the models in our run, labeled with our names; only where both values exist."""
    points = []
    for m in run.models:
        px = paper_value(claims, run.benchmark, m.name, x_metric, figure)
        py = paper_value(claims, run.benchmark, m.name, y_metric, figure)
        if px is not None and py is not None:
            points.append(Point(m.name, px, py * y_scale))
    return points


def baseline(claims: Claims, metric: str) -> float | None:
    """Majority-answer baseline for a metric (claims of kind 'baseline', from the dataset's counts)."""
    question = metric.removesuffix("_accuracy")
    target = f"majority_baseline_{question}"
    return next((as_fraction(c) for c in claims.claims if c.kind == "baseline" and c.metric == target), None)


@dataclass(frozen=True)
class Bar:
    name: str
    text: float | None
    video: float | None
    text_ci: tuple[float, float] | None
    video_ci: tuple[float, float] | None
    paper_text: float | None
    paper_video: float | None


def text_vs_video(text_run: RunData, video_run: RunData, claims: Claims, only_paper_models: bool) -> list[Bar]:
    """Fig 5a: Q1 latent-risk accuracy (text) next to yes/no risk accuracy (video), per model in both runs."""
    video = {model_key(m.name): m for m in video_run.models}
    bars = []
    for m in text_run.models:
        v = video.get(model_key(m.name))
        pt = paper_value(claims, "text", m.name, "latent_risk_accuracy", "5a")
        pv = paper_value(claims, "video", m.name, "risk_accuracy", "5a")
        if v is None or (only_paper_models and (pt is None or pv is None)):
            continue
        et, ev = estimate(m, "text", "latent_risk_accuracy"), estimate(v, "video", "risk_accuracy")
        bars.append(Bar(m.name, et and et.value, ev and ev.value, et and ci(et.lo, et.hi), ev and ci(ev.lo, ev.hi), pt, pv))
    return bars


def ci(lo: float | None, hi: float | None) -> tuple[float, float] | None:
    return (lo, hi) if lo is not None and hi is not None else None


def colors(names: list[str]) -> dict[str, str]:
    """Model name -> color, by position in `names` (the Fig 4 config's model order)."""
    return {model_key(n): PALETTE[i] if i < len(PALETTE) else FALLBACK_COLOR for i, n in enumerate(names)}


def same_spot_ranks(points: list[Point]) -> list[tuple[int, int]]:
    """(rank, count) per point among the points at the same (x, y), in input order.

    Small runs give identical scores, so several markers can sit on one spot; the plot rings them
    (rank 0 drawn largest) so every model's color stays visible without moving any point.
    """
    spot = [(round(p.x or 0.0, SAME_SPOT_DECIMALS), round(p.y or 0.0, SAME_SPOT_DECIMALS)) for p in points]
    return [(spot[:i].count(s), spot.count(s)) for i, s in enumerate(spot)]


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in display units (e.g. pixels): a label or a marker."""

    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: Box) -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1

    def shifted(self, dy: float) -> Box:
        return Box(self.x0, self.y0 + dy, self.x1, self.y1 + dy)

    def inside(self, bounds: Box) -> bool:
        return bounds.x0 <= self.x0 and self.x1 <= bounds.x1 and bounds.y0 <= self.y0 and self.y1 <= bounds.y1


def place_labels(options: list[list[Box]], obstacles: list[Box], step: float, bounds: Box) -> list[tuple[int, float]]:
    """(option index, vertical shift) per label, so labels stay inside `bounds` and clear of each other and the obstacles.

    Each label has home positions in order of preference (e.g. right of its marker, then left).
    Deterministic greedy: labels are placed in input order; each tries every option at its home
    height, then one step down, one up, two down, ..., and takes the first free slot. If none is
    free it keeps its first option, unshifted.
    """
    placed: list[Box] = []
    chosen = []
    shifts = [0.0] + [sign * k * step for k in range(1, MAX_LABEL_SHIFTS + 1) for sign in (-1, 1)]
    for boxes in options:
        slots = ((i, dy) for dy in shifts for i in range(len(boxes)))
        free = (
            (i, dy)
            for i, dy in slots
            if boxes[i].shifted(dy).inside(bounds) and not any(boxes[i].shifted(dy).overlaps(t) for t in placed + obstacles)
        )
        i, dy = next(free, (0, 0.0))
        placed.append(boxes[i].shifted(dy))
        chosen.append((i, dy))
    return chosen
