#!/usr/bin/env python
"""Draw the paper's figures from our results (docs/PLAN.md §10). Free: reads results/ and configs/ only.

    python scripts/plot.py                       # every run -> figures/<run_id>/<fig>_{paper,comparison}.{png,pdf}
    python scripts/plot.py --severity binary     # Fig 4 severity axis: 4way (default) | binary | recall
                                                 #   -> figures/<run_id>/fig4*_<version>_severity-binary.*

Two versions per figure:
- paper:      the paper's look (chart type, axes, colors, labels, legend; no title), our values only;
- comparison: ours filled with 95% CI bars, the paper's values hollow, majority baselines dashed.
Every run gets its own figures, under its run ID (results/run_ids.yaml): text runs Fig 4a and 4b, video runs Fig 5b.
Fig 5a (text vs video) pairs each run with the headline run of the other benchmark (results/headline.yaml) and is
skipped when there is none. Run pages (scripts/collect.py) link these files; Fig 6 waits for the image scorer.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from math import isnan, nan
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402
from matplotlib.typing import RcKeyType  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.bench.text_score import QUESTIONS  # noqa: E402
from asimov.claims import CLAIMS_PATH, Claims, load_claims, model_key  # noqa: E402
from asimov.collect.build import default_headline  # noqa: E402
from asimov.collect.load import RunData, load_results, load_yaml  # noqa: E402
from asimov.collect.pages import PAIRED_FIGURE, fig5a_partner, run_figures  # noqa: E402
from asimov.collect.run_ids import RUN_IDS_FILE, assign_ids, parse_ids  # noqa: E402
from asimov.plot_data import (  # noqa: E402
    SEVERITY,
    SEVERITY_LABEL,
    Bar,
    Box,
    Point,
    baseline,
    color,
    in_paper_order,
    needs_leader,
    ours,
    paper,
    place_labels,
    same_spot_ranks,
    severity_metric,
    text_vs_video,
)

# The paper's look: seaborn "darkgrid" (grey panel, white grid, no tick marks), without seaborn itself
STYLE: dict[RcKeyType, Any] = {
    "axes.facecolor": "#eaeaf2",
    "axes.edgecolor": "white",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}
# Paper-style figures copy the paper's larger type
PAPER_STYLE: dict[RcKeyType, Any] = {
    **STYLE,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
}
INK, MUTED = "#2b2b2b", "#6b6b6b"
CI_ALPHA = 0.6  # CI bars sit behind the dots
CONNECTOR = {"lw": 0.8, "alpha": 0.5, "zorder": 1}  # line from our dot to the paper's
BASELINE_LINE = {"color": MUTED, "ls": "--", "lw": 1, "zorder": 1}
PAPER_MARKER_SIZE = 40  # Fig 5a diamonds, points^2
RING_WIDTH = 6.0  # points of diameter added per extra model on the same spot, so each shows as a ring
AXIS_MARGIN = 0.07  # fraction of the data range (CI bars included) kept free on every side for labels
LABEL_PAD = 2.5  # points between a marker's edge and its label
LABEL_STEP = 0.25  # label heights per placement step: labels move in small steps so they stay near their dot
DIAGONALS = ("below right", "below left", "above right", "above left")  # fallback homes after the look's own sides
LEADER = {"arrowstyle": "-", "color": MUTED, "lw": 0.5}  # line from a moved label back to its marker
OUTSIDE_LEGEND = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0), "borderaxespad": 0.0, "frameon": True}
LEGEND_MARKER = 8.0  # points
POINTS_PER_INCH = 72
HIGHER, LOWER_INVERTED = " (Higher is Better)", " (Lower is Better, Inverted axis)"
OURS_NOTE, PAPER_NOTE = "filled = ours (95% CI)", "hollow = paper"


@dataclass(frozen=True)
class Look:
    """How a scatter figure is drawn: the paper's look, or the comparison look with room for CI bars."""

    figsize: tuple[float, float]  # inches
    marker_diameter: float  # points
    marker_alpha: float
    label_fontsize: float
    label_box: dict  # background behind each label
    legend: dict  # matplotlib legend placement
    legend_title: str | None
    sides: tuple[str, ...]  # label positions around the marker, preferred first


# Fig 4 in the paper: big translucent dots, name below in a white box, "Model" legend inside (top left in the
# paper; "best" here so it never covers a point, which puts it in the same empty corner when the data allow)
PAPER_FIG4 = Look(
    (10, 7),
    20.0,
    0.8,
    13,
    {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.6},
    {"loc": "upper left", "frameon": True},
    "Model",
    ("below", "above", "right", "left", *DIAGONALS),
)
# Fig 5b in the paper: same dots, name to the right, untitled legend inside (bottom right in the paper)
PAPER_FIG5B = Look(
    PAPER_FIG4.figsize,
    PAPER_FIG4.marker_diameter,
    PAPER_FIG4.marker_alpha,
    PAPER_FIG4.label_fontsize,
    PAPER_FIG4.label_box,
    {"loc": "lower right", "frameon": True},
    None,
    ("right", "left", "below", "above", *DIAGONALS),
)
# Comparison: smaller dots so CI bars and the paper's hollow dots stay readable; legend outside
COMPARISON = Look(
    (8.5, 5),
    8.4,
    1.0,
    7,
    {"boxstyle": "round,pad=0.15", "facecolor": STYLE["axes.facecolor"], "edgecolor": "none", "alpha": 0.85},
    OUTSIDE_LEGEND,
    "Model",
    ("right", "left", *DIAGONALS),
)
BAR_WIDTH = 0.38
BAR_VALUE_RISE = 2  # points above the bar (or its CI bar) for the printed value
BAR_HEADROOM = 0.06  # above the tallest bar or CI, for its printed value
PAPER_BAR_YMAX = 1.0  # the paper's Fig 5a y axis
FIG5A_INCHES_PER_MODEL, FIG5A_EXTRA_WIDTH, FIG5A_MIN_WIDTH, FIG5A_HEIGHT = 1.3, 2.5, 5.0, 4.5
PAPER_FIG5A_SIZE = (10.5, 7)
TICK_FONTSIZE = 10
COMPARISON_VALUE_FONTSIZE, PAPER_VALUE_FONTSIZE = 7, 13
# Fig 5a's two series: seaborn "deep" blue and orange at barplot saturation 0.75, as in the paper
TEXT_COLOR, VIDEO_COLOR = "#5975a4", "#cc8963"
FIG4 = {  # question, y metric, x axis name, y axis title, caption
    "fig4a": (
        QUESTIONS[1],  # latent severity on x
        f"{QUESTIONS[0]}_accuracy",  # latent risk type on y
        "Severity Accuracy",
        "Risk Recognition Accuracy" + HIGHER,
        "Perceiving Latent Risks Types and Severity",
    ),
    "fig4b": (
        QUESTIONS[3],  # activated risk on x
        f"{QUESTIONS[2]}_accuracy",  # action effect on y
        "Activated Risk Accuracy",
        "Action Effect Accuracy" + HIGHER,
        "Understanding Action Effects",
    ),
}


# Comparison figures use plain-language axis titles and percent ticks (Noa, 2026-10-12); the paper-style
# figures keep the paper's own wording. One entry per metric a comparison axis can show.
SCENARIOS = "\n% of scenarios, higher is better"  # second, smaller-looking line keeps titles short
READABLE = {
    f"{QUESTIONS[0]}_accuracy": "Picks the right injury type" + SCENARIOS,
    f"{QUESTIONS[1]}_accuracy": "Picks the right severity level" + SCENARIOS,
    f"{QUESTIONS[1]}_severe_binary_accuracy": "Tells severe injuries from the rest" + SCENARIOS,
    f"{QUESTIONS[1]}_severe_recall": "Catches the truly severe injuries\n% of severe scenarios, higher is better",
    f"{QUESTIONS[2]}_accuracy": "Judges how the action changes the risk" + SCENARIOS,
    f"{QUESTIONS[3]}_accuracy": "Picks the right severity after the action" + SCENARIOS,
    f"{QUESTIONS[3]}_severe_binary_accuracy": "After the action: tells severe from the rest" + SCENARIOS,
    f"{QUESTIONS[3]}_severe_recall": "After the action: catches the truly severe\n% of severe scenarios, higher is better",
    "risk_accuracy": "Recognizes whether there is an injury risk\n% correct, higher is better",
    "timing_mae_s": "Average timing error in seconds\nlower is better (axis reversed: better is to the right)",
    "within_0_5_rate": "Steps in within 0.5 s of the right moment\n% of videos, higher is better",
}
PERCENT_STEP = 0.1  # one tick every 10 points on fraction axes


def percent_axis(axis, scale: float = 1.0) -> None:
    """Ticks as whole percents every 10 points; `scale` is the axis value of 100% (1 for fractions, 100 for percents)."""
    axis.set_major_locator(MultipleLocator(PERCENT_STEP * scale))
    axis.set_major_formatter(PercentFormatter(xmax=scale, decimals=0))


# "How to read this" box under the legend of each comparison figure: what the marks mean, never what the
# results say (interpretation stays with the owners, CLAUDE.md).
NOTE_WIDTH = 40  # characters per line
NOTE_GAP = 0.03  # axes fraction between the legend and the box
NOTE_BOX = {"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#cccccc"}
NOTE_FONTSIZE = 8.5
CI_NOTE = "Filled dot = our run. Its bars show the 95% range the true score likely falls in."
PAPER_NOTE_LINE = "Hollow dot = the paper's value for the same model (same color)."
CONNECTOR_NOTE = " A thin line joins each model's two dots."
BASELINE_NOTE = "Dashed line = the score from always giving the most common answer."
FIG5A_NOTES = [
    "Each pair of bars is one model: the same risk question asked about text scenarios (blue) and videos (orange).",
    "Taller is better. The gap between a model's two bars is its text-vs-video difference.",
    "Bars = our runs, with the 95% range the true score likely falls in.",
    "Diamonds = the paper's values (only 3 models).",
]


def axis_meaning(metric: str) -> str:
    """First line of the plain-language axis title, lowercased: 'picks the right injury type'."""
    first = READABLE[metric].split("\n")[0]
    return first[0].lower() + first[1:]


def scatter_notes(x_metric: str, y_metric: str, reversed_x: bool = False, connected: bool = True) -> list[str]:
    right = (
        "Further right = smaller timing error (axis reversed)." if reversed_x else f"Further right = {axis_meaning(x_metric)}."
    )
    return [
        "Each dot is one model; up and to the right is better.",
        f"Higher = {axis_meaning(y_metric)}.",
        right,
        CI_NOTE,
        PAPER_NOTE_LINE + (CONNECTOR_NOTE if connected else ""),
    ]


def reading_notes(ax, lines: list[str]) -> None:
    """Draw a "How to read this" box just under the legend (outside the plot on the right)."""
    legend = ax.get_legend()
    fig = ax.figure
    fig.canvas.draw()
    if legend is not None:
        box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        x, y = box.x0, box.y0 - NOTE_GAP
    else:
        x, y = OUTSIDE_LEGEND["bbox_to_anchor"][0], 1.0
    wrapped = ["\n".join(textwrap.wrap(line, NOTE_WIDTH, subsequent_indent="  ")) for line in lines]
    text = "How to read this\n" + "\n".join("• " + w for w in wrapped)
    ax.text(x, y, text, transform=ax.transAxes, va="top", ha="left", fontsize=NOTE_FONTSIZE, bbox=NOTE_BOX, color=INK)


def marker_sizes(points: list[Point], diameter: float) -> list[float]:
    """Area (points^2) per marker: models sharing a spot get growing rings, the first one drawn largest."""
    return [(diameter + RING_WIDTH * (count - 1 - rank)) ** 2 for rank, count in same_spot_ranks(points)]


def scatter(ax, points: list[Point], look: Look, hollow: bool = False, with_ci: bool = False) -> None:
    for p, size in zip(points, marker_sizes(points, look.marker_diameter), strict=True):
        c = color(p.name)
        if with_ci and (p.x_ci or p.y_ci):
            xerr = [[p.x - p.x_ci[0]], [p.x_ci[1] - p.x]] if p.x_ci and p.x is not None else None
            yerr = [[p.y - p.y_ci[0]], [p.y_ci[1] - p.y]] if p.y_ci and p.y is not None else None
            ax.errorbar(p.x, p.y, xerr=xerr, yerr=yerr, fmt="none", ecolor=c, elinewidth=1, alpha=CI_ALPHA, zorder=2)
        face, edge = ("white", c) if hollow else (c, "white")
        alpha = 1.0 if hollow else look.marker_alpha
        ax.scatter(p.x, p.y, s=size, facecolors=face, edgecolors=edge, linewidths=1.5, alpha=alpha, zorder=3 + 1 / size)


def label_points(ax, points: list[Point], look: Look, avoid: list[Point] | None = None) -> None:
    """Name every point; a label that would overlap another label, a marker or the legend, or leave the plot,
    moves to the nearest free spot around its marker (a leader line when it ends up away from it).

    `avoid`: other dots labels must keep clear of (the paper's hollow dots on comparison figures).
    Call after the axis limits are final (inverted axes) and after the legend; placement is measured in display pixels.
    """
    ax.margins(AXIS_MARGIN)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_point = fig.dpi / POINTS_PER_INCH
    sizes = marker_sizes(points, look.marker_diameter)
    order = sorted(range(len(points)), key=lambda i: (-(points[i].y or 0.0), points[i].x or 0.0))  # top to bottom
    homes = [label_homes(sizes[i], look) for i in order]
    options = [[measure(ax, points[i], home, look, renderer) for home in homes[k]] for k, i in enumerate(order)]
    markers = [marker_box(ax, p, size, px_per_point) for p, size in zip(points, sizes, strict=True)]
    others = avoid or []
    markers += [
        marker_box(ax, p, size, px_per_point) for p, size in zip(others, marker_sizes(others, look.marker_diameter), strict=True)
    ]
    legend = ax.get_legend()
    if legend is not None:  # draw the legend first, so labels keep clear of it
        markers.append(box_of(legend.get_window_extent(renderer)))
    step = max((b.y1 - b.y0 for boxes in options for b in boxes), default=0.0) * LABEL_STEP
    own = [markers[i] for i in order]
    placements = place_labels(options, markers, step, axes_box(ax), own)
    for k, (i, (choice, dx_px, dy_px)) in enumerate(zip(order, placements, strict=True)):
        (dx, dy), ha, va = homes[k][choice]
        moved = needs_leader(options[k][choice].shifted(dx_px, dy_px), own[k])
        offset = (dx + dx_px / px_per_point, dy + dy_px / px_per_point)
        annotate(ax, points[i], offset, ha, va, look, LEADER if moved else None)


Home = tuple[tuple[float, float], str, str]  # offset from the point (points), horizontal and vertical alignment


def label_homes(size: float, look: Look) -> list[Home]:
    """Label positions around a marker of area `size`, in the look's order of preference."""
    gap = size**0.5 / 2 + LABEL_PAD
    diagonal = gap / 2**0.5  # the label's corner sits on the same circle as the side homes
    sides: dict[str, Home] = {
        "right": ((gap, 0.0), "left", "center"),
        "left": ((-gap, 0.0), "right", "center"),
        "below": ((0.0, -gap), "center", "top"),
        "above": ((0.0, gap), "center", "bottom"),
        "below right": ((diagonal, -diagonal), "left", "top"),
        "below left": ((-diagonal, -diagonal), "right", "top"),
        "above right": ((diagonal, diagonal), "left", "bottom"),
        "above left": ((-diagonal, diagonal), "right", "bottom"),
    }
    return [sides[s] for s in look.sides]


def measure(ax, p: Point, home: Home, look: Look, renderer) -> Box:
    draft = annotate(ax, p, *home, look)
    box = box_of(draft.get_window_extent(renderer))
    draft.remove()
    return box


def annotate(ax, p: Point, offset: tuple[float, float], ha: str, va: str, look: Look, leader: dict | None = None):
    return ax.annotate(
        p.name,
        (p.x, p.y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=look.label_fontsize,
        color="black",
        arrowprops=leader,
        bbox=look.label_box,
        zorder=4,
    )


def box_of(extent) -> Box:
    return Box(extent.x0, extent.y0, extent.x1, extent.y1)


def axes_box(ax) -> Box:
    return box_of(ax.get_window_extent())


def marker_box(ax, p: Point, size: float, px_per_point: float) -> Box:
    cx, cy = ax.transData.transform((p.x, p.y))
    r = size**0.5 / 2 * px_per_point
    return Box(cx - r, cy - r, cx + r, cy + r)


def model_legend(ax, points: list[Point], look: Look) -> None:
    """One legend entry per model in `points`, in the paper's legend order; none if there are no points.

    Inside the plot as in the paper; moved just outside when no spot inside is clear of every marker.
    """
    if not points:
        return
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markersize=LEGEND_MARKER,
            markerfacecolor=color(p.name),
            markeredgecolor="white",
            label=p.name,
        )
        for p in in_paper_order(points)
    ]
    legend = ax.legend(handles=handles, title=look.legend_title, **look.legend)
    if look.legend is not OUTSIDE_LEGEND and legend_covers(ax, legend, marker_boxes(ax, points, look)):
        ax.legend(handles=handles, title=look.legend_title, **OUTSIDE_LEGEND)


def marker_boxes(ax, points: list[Point], look: Look) -> list[Box]:
    px_per_point = ax.figure.dpi / POINTS_PER_INCH
    sizes = marker_sizes(points, look.marker_diameter)
    return [marker_box(ax, p, size, px_per_point) for p, size in zip(points, sizes, strict=True)]


def legend_covers(ax, legend, boxes: list[Box]) -> bool:
    """Whether the legend, where matplotlib put it, overlaps any of `boxes` (display pixels)."""
    fig = ax.figure
    fig.canvas.draw()
    box = box_of(legend.get_window_extent(fig.canvas.get_renderer()))
    return any(box.overlaps(b) for b in boxes)


def save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.{ext}" for ext in ("png", "pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def fig4(name: str, run: RunData, claims: Claims, severity: str, out: Path) -> list[Path]:
    question, y_metric, x_label, y_label, title = FIG4[name]
    x_metric = severity_metric(question, severity)
    x_label += SEVERITY_LABEL[severity] + HIGHER
    suffix = "" if severity == "4way" else f"_severity-{severity}"
    mine = ours(run, x_metric, y_metric)
    written = []
    with plt.rc_context(PAPER_STYLE):  # the paper prints no title: its caption sits under the figure
        fig, ax = plt.subplots(figsize=PAPER_FIG4.figsize)
        scatter(ax, mine, PAPER_FIG4)
        ax.set(xlabel=x_label, ylabel=y_label)
        model_legend(ax, mine, PAPER_FIG4)
        label_points(ax, mine, PAPER_FIG4)
        written += save(fig, out, f"{name}_paper{suffix}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=COMPARISON.figsize)
        theirs = paper(run, claims, name[3:], x_metric, y_metric)  # empty for binary/recall: the paper plots 4-way
        for p in theirs:
            m = next((q for q in mine if model_key(q.name) == model_key(p.name)), None)
            if m and m.x is not None and m.y is not None and p.x is not None and p.y is not None:
                ax.plot([m.x, p.x], [m.y, p.y], color=color(p.name), **CONNECTOR)
        scatter(ax, mine, COMPARISON, with_ci=True)
        scatter(ax, theirs, COMPARISON, hollow=True)
        draw_baselines(ax, claims, x_metric, y_metric)
        note = f"{OURS_NOTE}, {PAPER_NOTE}" if theirs else f"{OURS_NOTE}; no paper values for this axis"
        ax.set(xlabel=READABLE[x_metric], ylabel=READABLE[y_metric], title=f"{title}\n{note}")
        percent_axis(ax.xaxis)
        percent_axis(ax.yaxis)
        model_legend(ax, mine, COMPARISON)
        label_points(ax, mine, COMPARISON, avoid=theirs)
        has_baseline = baseline(claims, x_metric) is not None or baseline(claims, y_metric) is not None
        reading_notes(ax, scatter_notes(x_metric, y_metric) + ([BASELINE_NOTE] if has_baseline else []))
        written += save(fig, out, f"{name}_comparison{suffix}")
    return written


def draw_baselines(ax, claims: Claims, x_metric: str, y_metric: str) -> None:
    bx, by = baseline(claims, x_metric), baseline(claims, y_metric)
    if bx is not None:
        ax.axvline(bx, **BASELINE_LINE)
        ax.annotate(
            "majority answer",
            (bx, 0),
            xycoords=("data", "axes fraction"),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=COMPARISON.label_fontsize,
            color=MUTED,
        )
    if by is not None:
        ax.axhline(by, **BASELINE_LINE)
        ax.annotate(
            "majority answer",
            (0, by),
            xycoords=("axes fraction", "data"),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=COMPARISON.label_fontsize,
            color=MUTED,
        )


def bar_values(ax, xs, values: list[float | None], tops: list[float], fontsize: float) -> None:
    """Print each value above its bar, or above its CI bar when there is one (`tops`)."""
    for x, v, top in zip(xs, values, tops, strict=True):
        if v is not None:
            ax.annotate(
                f"{v:.2f}",
                (x, top),
                xytext=(0, BAR_VALUE_RISE),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=INK,
            )


TICK_WRAP = 11  # characters per line of a Fig 5a model name


def tick_name(name: str) -> str:
    """Model name wrapped to short lines, e.g. "Gemini 2.5\nFlash-Lite\n(3.1)", so Fig 5a's ticks stay horizontal
    and don't run into each other."""
    base, _, stand_in = name.partition(" (")
    lines = textwrap.wrap(base, TICK_WRAP, break_on_hyphens=False)
    return "\n".join(lines + ([f"({stand_in}"] if stand_in else []))


def fig5a(bars: list[Bar], stem: str, out: Path, comparison: bool) -> list[Path]:
    """Grouped bars, text vs video per model. Paper version: the paper's look; comparison: CI bars + paper diamonds."""
    bars = in_paper_order(bars)
    width = max(FIG5A_MIN_WIDTH, FIG5A_INCHES_PER_MODEL * len(bars) + FIG5A_EXTRA_WIDTH)
    size = (width, FIG5A_HEIGHT) if comparison else PAPER_FIG5A_SIZE
    value_size = COMPARISON_VALUE_FONTSIZE if comparison else PAPER_VALUE_FONTSIZE
    with plt.rc_context(STYLE if comparison else PAPER_STYLE):
        fig, ax = plt.subplots(figsize=size)
        xs = list(range(len(bars)))
        tallest = 0.0
        for series, (offset, kind, c) in enumerate(((-BAR_WIDTH / 2, "text", TEXT_COLOR), (BAR_WIDTH / 2, "video", VIDEO_COLOR))):
            pos = [x + offset for x in xs]
            values = [nan if getattr(b, kind) is None else getattr(b, kind) for b in bars]  # missing: no bar, not a 0
            cis = [getattr(b, f"{kind}_ci") if comparison else None for b in bars]
            err = (
                [
                    [v - ci[0] if ci else 0 for v, ci in zip(values, cis, strict=True)],
                    [ci[1] - v if ci else 0 for v, ci in zip(values, cis, strict=True)],
                ]
                if comparison
                else None
            )
            ax.bar(
                pos,
                values,
                BAR_WIDTH,
                color=c,
                edgecolor="white",
                linewidth=1,
                label=kind.capitalize(),
                yerr=err,
                error_kw={"elinewidth": 1, "ecolor": INK, "alpha": CI_ALPHA},
            )
            tops = [ci[1] if ci else v for v, ci in zip(values, cis, strict=True)]
            tallest = max([tallest, *(t for t in tops if not isnan(t))])
            bar_values(ax, pos, [getattr(b, kind) for b in bars], tops, value_size)
            if comparison:
                paper_bars(ax, pos, [getattr(b, f"paper_{kind}") for b in bars], first=series == 0)
        names = [tick_name(b.name) for b in bars]
        ax.set_xticks(xs, names, fontsize=TICK_FONTSIZE) if comparison else ax.set_xticks(xs, names)
        ylabel = READABLE["risk_accuracy"] if comparison else "Risk Recognition Accuracy"
        ax.set(xlabel="Model", ylabel=ylabel, ylim=(0, max(PAPER_BAR_YMAX, tallest + BAR_HEADROOM)))
        if comparison:
            ax.set_title(f"Text vs video\n{OURS_NOTE}, diamonds = paper")
            percent_axis(ax.yaxis)
        if bars:
            legend = ax.legend(title="Accuracy Type", **(OUTSIDE_LEGEND if comparison else {"loc": "best", "frameon": True}))
            drawn = [box_of(a.get_window_extent()) for a in [*ax.patches, *ax.texts]]
            if not comparison and legend_covers(ax, legend, drawn):  # the paper's corner is taken: move it out
                ax.legend(title="Accuracy Type", **OUTSIDE_LEGEND)
        if comparison and bars:
            reading_notes(ax, FIG5A_NOTES)
        return save(fig, out, stem)


def paper_bars(ax, pos: list[float], values: list[float | None], first: bool) -> None:
    """The paper's values as white diamonds on our bars; one "Paper" legend entry for both series."""
    shown = [(p, v) for p, v in zip(pos, values, strict=True) if v is not None]
    ax.scatter(
        [p for p, _ in shown],
        [v for _, v in shown],
        marker="D",
        s=PAPER_MARKER_SIZE,
        facecolors="white",
        edgecolors=INK,
        zorder=3,
        label="Paper" if first else None,
    )


def fig5b(run: RunData, claims: Claims, out: Path) -> list[Path]:
    x_label = "Last Time of Intervention Mean Absolute Error" + LOWER_INVERTED
    y_label, title = "Ratio of Accurate Interventions" + HIGHER, "Triggering Interventions"
    mine = ours(run, "timing_mae_s", "within_0_5_rate", y_scale=100)
    theirs = paper(run, claims, "5b", "timing_mae_s", "within_0_5_rate", y_scale=100)
    written = []
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=PAPER_FIG5B.figsize)
        scatter(ax, mine, PAPER_FIG5B)
        ax.invert_xaxis()
        ax.set(xlabel=x_label, ylabel=y_label)
        model_legend(ax, mine, PAPER_FIG5B)
        label_points(ax, mine, PAPER_FIG5B)
        written += save(fig, out, "fig5b_paper")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=COMPARISON.figsize)
        scatter(ax, mine, COMPARISON, with_ci=True)
        scatter(ax, theirs, COMPARISON, hollow=True)
        ax.invert_xaxis()
        ax.set(xlabel=READABLE["timing_mae_s"], ylabel=READABLE["within_0_5_rate"], title=f"{title}\n{OURS_NOTE}, {PAPER_NOTE}")
        percent_axis(ax.yaxis, scale=100)
        model_legend(ax, mine, COMPARISON)
        label_points(ax, mine, COMPARISON, avoid=theirs)
        reading_notes(ax, scatter_notes("timing_mae_s", "within_0_5_rate", reversed_x=True, connected=False))
        written += save(fig, out, "fig5b_comparison")
    return written


def draw_run(run: RunData, runs: list[RunData], headline: dict, claims: Claims, severity: str, out: Path) -> list[Path]:
    """This run's figures (pages.run_figures) into `out`, which is figures/<run_id>/."""
    names = run_figures(run, runs, headline)
    written: list[Path] = []
    for name in (n for n in names if n in FIG4):
        written += fig4(name, run, claims, severity, out)
    partner = fig5a_partner(run, runs, headline)
    if PAIRED_FIGURE in names and partner is not None:
        text, video = (run, partner) if run.benchmark == "text" else (partner, run)
        written += fig5a(text_vs_video(text, video, claims, only_paper_models=True), "fig5a_paper", out, False)
        written += fig5a(text_vs_video(text, video, claims, only_paper_models=False), "fig5a_comparison", out, True)
    if "fig5b" in names:
        written += fig5b(run, claims, out)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--severity", choices=sorted(SEVERITY), default="4way")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--claims", type=Path, default=CLAIMS_PATH)
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args(argv)

    runs = load_results(args.results)
    claims = load_claims(args.claims)
    headline = load_yaml(args.results / "headline.yaml", {}) or default_headline(runs)  # as collect.py reads it
    ids = assign_ids(parse_ids(load_yaml(args.results / RUN_IDS_FILE, {})), runs)  # new runs: the IDs collect.py gives
    written: list[Path] = []
    for run in runs:
        written += draw_run(run, runs, headline, claims, args.severity, args.out / ids[run.name])
    print("fig6: not drawn yet, waiting for the image scorer (Riddhi)")
    unpaired = [ids[r.name] for r in runs if r.benchmark in ("text", "video") and fig5a_partner(r, runs, headline) is None]
    if unpaired:
        print(f"fig5a: skipped for {', '.join(unpaired)}: no headline run of the other benchmark (results/headline.yaml)")
    print(f"plot: {len(written)} files for {len(runs)} runs under {args.out}/<run_id>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
