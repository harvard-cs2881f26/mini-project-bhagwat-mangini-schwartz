"""What each paper figure plots (docs/PLAN.md §10), as plain data: pure functions, no matplotlib.

Values and CIs come from asimov.collect.metrics.estimate, the same code the results pages use, so
a figure and its page can't disagree. Paper values come from configs/paper/claims.yaml.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Protocol, TypeVar

from asimov.claims import Claims, as_fraction, model_key, paper_value
from asimov.collect.load import RunData
from asimov.collect.metrics import estimate

# Fig 4a/4b severity axis: DECIDED, 4-way (multiple choice, exact level right), WORKLOG 2026-10-12 (Noa).
# The other two stay available as `scripts/plot.py --severity binary|recall` for comparison.
SEVERITY = {"4way": "_accuracy", "binary": "_severe_binary_accuracy", "recall": "_severe_recall"}
# Goes between the axis name and "(Higher is Better)", e.g. "Severity Accuracy, severe vs not (Higher is Better)".
SEVERITY_LABEL = {"4way": "", "binary": ", severe vs not", "recall": ", severe recall"}
# The paper's colors (seaborn "deep"), one per model as in its Fig 4 legend, in that legend's order.
# Used in every figure so ours sit beside the paper's on a slide (Noa, 2026-10-12). The paper's own
# Fig 5b legend swaps GPT 5 Nano and Mini; we keep the Fig 4 colors everywhere.
PAPER_COLORS = {
    "Gemini 2.5 Flash-Lite": "#4c72b0",
    "Gemini 2.5 Flash": "#dd8452",
    "Gemini 2.5 Pro": "#55a868",
    "Claude Opus 4.1": "#c44e52",
    "Claude Sonnet 4": "#8172b3",
    "GPT 5 Nano": "#937860",
    "GPT 5 Mini": "#da8bc3",
    "GPT 5": "#8c8c8c",
}
FALLBACK_COLOR = "#6b6b6b"
SAME_SPOT_DECIMALS = 6  # points equal to this many decimals are drawn as one spot (ringed markers)
MAX_LABEL_SHIFTS = 48  # grid steps tried each way (left, right, down, up) from a label's home before it is left there
SIDE_SWITCH_COST = 0.5  # label heights: a free spot this much farther on the preferred side still beats another side
CROSSING_COST = 2.0  # label heights added per leader-line crossing, or for a label nearer another marker than its own
LEADER_GAP = 0.5  # label heights between a label and its marker beyond which a leader line is drawn
AMBIGUITY_MARGIN = 1.0  # label heights: another marker this much farther than a label's own still makes it ambiguous
FAR_MOVE = 2.0  # label heights: a move this long costs double its distance
IMPROVE_ROUNDS = 5  # passes that re-place each label given all the others, after the first greedy pass


@dataclass(frozen=True)
class Point:
    name: str  # our config name, e.g. "Claude Opus 4.1 (4.5)"
    x: float | None
    y: float | None
    x_ci: tuple[float, float] | None = None
    y_ci: tuple[float, float] | None = None


def severity_metric(question: str, severity: str) -> str:
    return question + SEVERITY[severity]


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


def color(name: str) -> str:
    """The paper's color for a model (our stand-in names map to the paper model they stand in for)."""
    return next((c for n, c in PAPER_COLORS.items() if model_key(n) == model_key(name)), FALLBACK_COLOR)


class Named(Protocol):
    @property
    def name(self) -> str: ...


N = TypeVar("N", bound=Named)


def in_paper_order(items: list[N]) -> list[N]:
    """Points or bars in the paper's legend order; models the paper doesn't have go last, by name."""
    rank = {model_key(n): i for i, n in enumerate(PAPER_COLORS)}
    return sorted(items, key=lambda p: (rank.get(model_key(p.name), len(rank)), p.name))


def same_spot_ranks(points: list[Point]) -> list[tuple[int, int]]:
    """(rank, count) per point among the points at the same (x, y), in input order.

    Small runs give identical scores, so several markers can sit on one spot; the plot rings them
    (rank 0 drawn largest) so every model's color stays visible without moving any point.
    """
    spot = [(round(p.x or 0.0, SAME_SPOT_DECIMALS), round(p.y or 0.0, SAME_SPOT_DECIMALS)) for p in points]
    return [(spot[:i].count(s), spot.count(s)) for i, s in enumerate(spot)]


XY = tuple[float, float]


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in display units (e.g. pixels): a label or a marker."""

    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: Box) -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1

    def shifted(self, dx: float, dy: float) -> Box:
        return Box(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def inside(self, bounds: Box) -> bool:
        return bounds.x0 <= self.x0 and self.x1 <= bounds.x1 and bounds.y0 <= self.y0 and self.y1 <= bounds.y1

    def center(self) -> XY:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def distance(self, other: Box) -> float:
        """Gap between the two boxes; 0 when they touch or overlap."""
        gap_x = max(other.x0 - self.x1, self.x0 - other.x1, 0.0)
        gap_y = max(other.y0 - self.y1, self.y0 - other.y1, 0.0)
        return math.hypot(gap_x, gap_y)

    def crossed_by(self, a: XY, b: XY) -> bool:
        """Whether the segment a-b passes through the box's interior (Liang-Barsky clipping)."""
        t0, t1 = 0.0, 1.0
        edges = (
            (a[0] - b[0], a[0] - self.x0),
            (b[0] - a[0], self.x1 - a[0]),
            (a[1] - b[1], a[1] - self.y0),
            (b[1] - a[1], self.y1 - a[1]),
        )
        for p, q in edges:
            if p == 0:
                if q <= 0:
                    return False
            elif p < 0:
                t0 = max(t0, q / p)
            else:
                t1 = min(t1, q / p)
        return t0 < t1


Segment = tuple[XY, XY]
Placement = tuple[int, float, float]  # option index, horizontal shift, vertical shift


def segments_cross(s: Segment, t: Segment) -> bool:
    """Whether two segments properly cross (touching at an end doesn't count)."""

    def side(a: XY, b: XY, c: XY) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    (a, b), (c, d) = s, t
    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def needs_leader(label: Box, marker: Box) -> bool:
    """A label more than LEADER_GAP of its own height away from its marker gets a line back to it."""
    return label.distance(marker) > LEADER_GAP * (label.y1 - label.y0)


def leader(label: Box, marker: Box) -> Segment:
    """The line from the marker's center toward its label's center (matplotlib draws it that way, clipped at the label)."""
    return (marker.center(), label.center())


@cache
def candidates(count: int, height: float, step: float) -> tuple[tuple[float, Placement], ...]:
    """Every (option, dx, dy) on the step grid with its cost, cheapest first.

    Cost is `move_cost`. Ties go to the earlier option, then to the smaller sideways move, then the smaller
    vertical one, down before up.
    """
    reach = [k * step for k in range(-MAX_LABEL_SHIFTS, MAX_LABEL_SHIFTS + 1)]
    grid = [(move_cost((i, dx, dy), height), (i, dx, dy)) for i in range(count) for dx in reach for dy in reach]
    return tuple(sorted(grid, key=lambda c: (c[0], c[1][0], abs(c[1][1]), abs(c[1][2]), c[1][2], c[1][1])))


@dataclass(frozen=True)
class Scene:
    """What one label must keep clear of: the other labels and leader lines, the obstacles and the plot edge."""

    labels: list[Box]
    lines: list[Segment]
    obstacles: list[Box]
    bounds: Box

    def free(self, box: Box) -> bool:
        return box.inside(self.bounds) and not any(box.overlaps(t) for t in self.labels + self.obstacles)


@dataclass(frozen=True)
class Spot:
    """A free spot for one label: its total cost, placement, box and leader line (None when beside its marker)."""

    cost: float
    placement: Placement
    box: Box
    line: Segment | None


def penalty(box: Box, line: Segment | None, scene: Scene, marker: Box, dots: list[Box]) -> int:
    """Crossings and ambiguity, in units of CROSSING_COST label heights.

    Counts the leader lines this label covers and, for its own leader line, the labels, leader lines and other
    markers it runs through. A label that is not clearly nearest its own marker (another marker is within
    AMBIGUITY_MARGIN label heights of as near) counts too.
    """
    hits = sum(box.crossed_by(*s) for s in scene.lines)
    if line is not None:
        hits += sum(segments_cross(line, s) for s in scene.lines) + sum(b.crossed_by(*line) for b in scene.labels + dots)
    margin = AMBIGUITY_MARGIN * (box.y1 - box.y0)
    if any(box.distance(d) < box.distance(marker) + margin for d in dots):
        hits += 1
    return hits


def spot(boxes: list[Box], placement: Placement, cost: float, scene: Scene, marker: Box | None, dots: list[Box]) -> Spot:
    """Label `boxes` at `placement`, with `cost` (its move) plus its penalty when its marker is known."""
    i, dx, dy = placement
    box = boxes[i].shifted(dx, dy)
    if marker is None:
        return Spot(cost, placement, box, None)
    line = leader(box, marker) if needs_leader(box, marker) else None
    return Spot(
        cost + CROSSING_COST * (boxes[0].y1 - boxes[0].y0) * penalty(box, line, scene, marker, dots), placement, box, line
    )


def best_spot(boxes: list[Box], scene: Scene, marker: Box | None, dots: list[Box], step: float) -> Spot | None:
    """The cheapest free spot for one label; candidates come cheapest first, so the search stops once no
    later one can beat the best found (penalties only add)."""
    best: Spot | None = None
    for cost, placement in candidates(len(boxes), boxes[0].y1 - boxes[0].y0, step):
        if best is not None and cost >= best.cost:
            break
        i, dx, dy = placement
        if scene.free(boxes[i].shifted(dx, dy)):
            found = spot(boxes, placement, cost, scene, marker, dots)
            best = found if best is None or found.cost < best.cost else best
    return best


def place_labels(
    options: list[list[Box]], obstacles: list[Box], step: float, bounds: Box, markers: list[Box] | None = None
) -> list[Placement]:
    """(option index, dx, dy) per label, so labels stay inside `bounds` and clear of each other and the obstacles.

    Each label has home positions in order of preference (e.g. below its marker, then above, then diagonals).
    A label takes the free spot on a `step` grid around its homes that is closest to home (see `candidates`).
    With `markers` (each label's own marker), a spot costs CROSSING_COST label heights more for each crossing
    of its leader line with a label, leader line or marker, and for sitting nearer another marker than its own,
    so a slightly farther clear spot wins.

    Deterministic: a greedy pass, then up to IMPROVE_ROUNDS passes that move each label in turn to its best spot
    given all the others, while that strictly lowers its cost. Without markers that runs once, in input order;
    with markers it runs for a few label orders (`orders`) and keeps the cheapest layout, the earliest on a tie.
    A label with no free spot keeps its first home, unshifted.
    """
    layouts = [Layout(options, obstacles, step, bounds, markers, order) for order in orders(len(options), markers)]
    best = min(layouts, key=lambda layout: layout.score())
    return [s.placement if s is not None else (0, 0.0, 0.0) for s in best.spots]


def orders(count: int, markers: list[Box] | None) -> list[list[int]]:
    """Label orders to try: as given (without markers, only that), reversed, and markers left to right and back."""
    given = list(range(count))
    if markers is None:
        return [given]
    by_x = sorted(given, key=lambda k: (markers[k].center()[0], k))
    found: list[list[int]] = []
    for order in (given, given[::-1], by_x, by_x[::-1]):
        if order not in found:
            found.append(order)
    return found


class Layout:
    """One deterministic placement of every label, in a given order (see `place_labels`)."""

    def __init__(
        self,
        options: list[list[Box]],
        obstacles: list[Box],
        step: float,
        bounds: Box,
        markers: list[Box] | None,
        order: list[int],
    ) -> None:
        self.options, self.obstacles, self.step, self.bounds, self.markers = options, obstacles, step, bounds, markers
        self.spots: list[Spot | None] = [None] * len(options)
        for k in order:
            self.spots[k] = best_spot(options[k], self.scene(k), *_own(markers, k), step)
        for _ in range(IMPROVE_ROUNDS if markers is not None else 0):
            moved = [self.improve(k) for k in order]  # every label gets its turn, in order
            if not any(moved):
                break

    def scene(self, k: int) -> Scene:
        """Everything label k must keep clear of: the other labels placed so far and their leader lines."""
        others = [s for j, s in enumerate(self.spots) if j != k and s is not None]
        return Scene([s.box for s in others], [s.line for s in others if s.line], self.obstacles, self.bounds)

    def current(self, k: int) -> Spot | None:
        """Label k's spot, costed against all the other labels as they are now."""
        held = self.spots[k]
        if held is None:
            return None
        boxes = self.options[k]
        return spot(
            boxes,
            held.placement,
            move_cost(held.placement, boxes[0].y1 - boxes[0].y0),
            self.scene(k),
            *_own(self.markers, k),
        )

    def improve(self, k: int) -> bool:
        """Move label k to its best spot given all the others, if that is strictly cheaper."""
        now = self.current(k)
        better = best_spot(self.options[k], self.scene(k), *_own(self.markers, k), self.step)
        if now is None or better is None or better.cost >= now.cost:
            return False
        self.spots[k] = better
        return True

    def score(self) -> tuple[int, float]:
        """(labels with no free spot, total cost of the others): lower is better."""
        costs = [self.current(k) for k in range(len(self.spots))]
        return sum(c is None for c in costs), sum(c.cost for c in costs if c is not None)


def move_cost(placement: Placement, height: float) -> float:
    """How far a label moved from its first home: the distance, growing faster once it passes FAR_MOVE label
    heights (so one long leader line costs more than a few short moves), plus SIDE_SWITCH_COST label heights
    for any other home."""
    i, dx, dy = placement
    distance = math.hypot(dx, dy)
    return distance * (1 + distance / (FAR_MOVE * height)) + (SIDE_SWITCH_COST * height if i else 0.0)


def _own(markers: list[Box] | None, k: int) -> tuple[Box | None, list[Box]]:
    """Label k's own marker and everyone else's."""
    if markers is None:
        return None, []
    return markers[k], [m for j, m in enumerate(markers) if j != k]
