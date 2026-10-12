#!/usr/bin/env python
"""Draw the paper's figures from our results (docs/PLAN.md §10). Free: reads results/ and configs/ only.

    python scripts/plot.py                       # Fig 4a, 4b, 5a, 5b -> figures/<fig>_{paper,comparison}.{png,pdf}
    python scripts/plot.py --severity binary     # Fig 4 severity axis: 4way (default) | binary | recall

Two versions per figure:
- paper:      same chart type, axes, titles and labels as the paper, our values only;
- comparison: ours filled with 95% CI bars, the paper's values hollow, majority baselines dashed.
Runs come from results/headline.yaml (the official run per figure). Fig 6 waits for the image scorer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.typing import RcKeyType  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.claims import CLAIMS_PATH, Claims, load_claims, model_key  # noqa: E402
from asimov.collect.build import default_headline  # noqa: E402
from asimov.collect.load import RunData, load_results, load_yaml  # noqa: E402
from asimov.plot_data import (  # noqa: E402
    SEVERITY,
    SEVERITY_LABEL,
    Bar,
    Box,
    Point,
    baseline,
    colors,
    ours,
    paper,
    place_labels,
    same_spot_ranks,
    severity_metric,
    text_vs_video,
)

# Paper look: seaborn-like grey panel with white grid (no seaborn dependency)
STYLE: dict[RcKeyType, Any] = {
    "axes.facecolor": "#eaeaf2",
    "axes.edgecolor": "white",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
}
INK, MUTED = "#2b2b2b", "#6b6b6b"
SCATTER_SIZE = (8.5, 5)  # inches; wider than the plot itself so the Model legend fits outside it
MARKER_DIAMETER = 8.4  # points
PAPER_MARKER_SIZE = 40  # Fig 5a diamonds, points^2
RING_WIDTH = 6.0  # points of diameter added per extra model on the same spot, so each shows as a ring
AXIS_MARGIN = 0.07  # fraction of the data range (CI bars included) kept free on every side for labels
LABEL_FONTSIZE = 7
LABEL_PAD = 2.5  # points between a marker's edge and its label
LABEL_FLIP_X = 0.7  # axes fraction right of which labels go to the left of their marker
# Label background in the panel color, so CI bars and connector lines don't strike through the text
LABEL_BOX = {"boxstyle": "round,pad=0.15", "facecolor": STYLE["axes.facecolor"], "edgecolor": "none", "alpha": 0.85}
LEADER = {"arrowstyle": "-", "color": MUTED, "lw": 0.5}  # line from a moved label back to its marker
OUTSIDE_LEGEND = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0), "borderaxespad": 0.0, "frameon": True}
POINTS_PER_INCH = 72
HIGHER, LOWER_INVERTED = " (Higher is Better)", " (Lower is Better, Inverted axis)"
OURS_NOTE, PAPER_NOTE = "filled = ours (95% CI)", "hollow = paper"
BAR_WIDTH = 0.38
BAR_VALUE_RISE = 2  # points above the bar (or its CI bar) for the printed value
BAR_YLIM = (0, 1.12)  # room above 1.0 for the printed values
FIG5A_INCHES_PER_MODEL, FIG5A_EXTRA_WIDTH, FIG5A_MIN_WIDTH, FIG5A_HEIGHT = 1.3, 2.5, 5.0, 4.5
TICK_FONTSIZE = 8
TEXT_COLOR, VIDEO_COLOR = "#2a78d6", "#eb6834"  # Fig 5a's two series (validated palette slots 1-2)
FIG4 = {  # question, y metric, x axis name, y axis title, caption
    "fig4a": (
        "latent_severity",
        "latent_risk_accuracy",
        "Severity Accuracy",
        "Risk Recognition Accuracy" + HIGHER,
        "Perceiving Latent Risks Types and Severity",
    ),
    "fig4b": (
        "activated_risk",
        "effect_accuracy",
        "Activated Risk Accuracy",
        "Action Effect Accuracy" + HIGHER,
        "Understanding Action Effects",
    ),
}


def marker_sizes(points: list[Point]) -> list[float]:
    """Area (points^2) per marker: models sharing a spot get growing rings, the first one drawn largest."""
    return [(MARKER_DIAMETER + RING_WIDTH * (count - 1 - rank)) ** 2 for rank, count in same_spot_ranks(points)]


def scatter(ax, points: list[Point], palette: dict[str, str], hollow: bool = False, with_ci: bool = False) -> None:
    for p, size in zip(points, marker_sizes(points), strict=True):
        color = palette.get(model_key(p.name), MUTED)
        if with_ci and (p.x_ci or p.y_ci):
            xerr = [[p.x - p.x_ci[0]], [p.x_ci[1] - p.x]] if p.x_ci and p.x is not None else None
            yerr = [[p.y - p.y_ci[0]], [p.y_ci[1] - p.y]] if p.y_ci and p.y is not None else None
            ax.errorbar(p.x, p.y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color, elinewidth=1, alpha=0.6, zorder=2)
        face = "white" if hollow else color
        edge = color if hollow else "white"
        ax.scatter(p.x, p.y, s=size, facecolors=face, edgecolors=edge, linewidths=1.5, zorder=3 + 1 / size)


def label_points(ax, points: list[Point]) -> None:
    """Name every point; a label that would overlap another label or a marker, or leave the plot, moves
    to the marker's other side or up/down (with a leader line).

    Call after the axis limits are final (inverted axes); placement is measured in display pixels.
    """
    ax.margins(AXIS_MARGIN)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_point = fig.dpi / POINTS_PER_INCH
    sizes = marker_sizes(points)
    order = sorted(range(len(points)), key=lambda i: (-(points[i].y or 0.0), points[i].x or 0.0))  # top to bottom
    homes = [label_homes(ax, points[i], sizes[i]) for i in order]
    options = [[measure(ax, points[i], home, renderer) for home in homes[k]] for k, i in enumerate(order)]
    markers = [marker_box(ax, p, size, px_per_point) for p, size in zip(points, sizes, strict=True)]
    step = max((b.y1 - b.y0 for boxes in options for b in boxes), default=0.0)
    for k, (i, (choice, shift_px)) in enumerate(zip(order, place_labels(options, markers, step, axes_box(ax)), strict=True)):
        (dx, dy), ha = homes[k][choice]
        shift = shift_px / px_per_point
        annotate(ax, points[i], (dx, dy + shift), ha, LEADER if shift else None)


Home = tuple[tuple[float, float], str]  # offset from the point (points) and horizontal alignment


def label_homes(ax, p: Point, size: float) -> list[Home]:
    """Label positions beside the marker, preferred first: right of it, or left near the right edge."""
    gap = size**0.5 / 2 + LABEL_PAD
    right, left = ((gap, 0.0), "left"), ((-gap, 0.0), "right")
    return [left, right] if ax.transLimits.transform((p.x, p.y))[0] > LABEL_FLIP_X else [right, left]


def measure(ax, p: Point, home: Home, renderer) -> Box:
    draft = annotate(ax, p, *home)
    box = box_of(draft.get_window_extent(renderer))
    draft.remove()
    return box


def annotate(ax, p: Point, offset: tuple[float, float], ha: str, leader: dict | None = None):
    return ax.annotate(
        p.name,
        (p.x, p.y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=LABEL_FONTSIZE,
        color=INK,
        arrowprops=leader,
        bbox=LABEL_BOX,
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


def model_legend(ax, points: list[Point], palette: dict[str, str]) -> None:
    """A "Model" legend outside the plot, one entry per model in `points`; none if there are no points."""
    if not points:
        return
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markersize=MARKER_DIAMETER,
            markerfacecolor=palette.get(model_key(p.name), MUTED),
            markeredgecolor="white",
            label=p.name,
        )
        for p in points
    ]
    ax.legend(handles=handles, title="Model", **OUTSIDE_LEGEND)


def save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.{ext}" for ext in ("png", "pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def fig4(name: str, run: RunData, claims: Claims, palette: dict[str, str], severity: str, out: Path) -> list[Path]:
    question, y_metric, x_label, y_label, title = FIG4[name]
    x_metric = severity_metric(question, severity)
    x_label += SEVERITY_LABEL[severity] + HIGHER
    suffix = "" if severity == "4way" else f"_severity-{severity}"
    mine = ours(run, x_metric, y_metric)
    written = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=SCATTER_SIZE)
        scatter(ax, mine, palette)
        label_points(ax, mine)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        model_legend(ax, mine, palette)
        written += save(fig, out, f"{name}_paper{suffix}")

        fig, ax = plt.subplots(figsize=SCATTER_SIZE)
        theirs = paper(run, claims, name[3:], x_metric, y_metric)  # empty for binary/recall: the paper plots 4-way
        for p in theirs:
            m = next((q for q in mine if model_key(q.name) == model_key(p.name)), None)
            if m and m.x is not None and m.y is not None and p.x is not None and p.y is not None:
                ax.plot([m.x, p.x], [m.y, p.y], color=palette.get(model_key(p.name), MUTED), lw=0.8, alpha=0.5, zorder=1)
        scatter(ax, mine, palette, with_ci=True)
        scatter(ax, theirs, palette, hollow=True)
        draw_baselines(ax, claims, x_metric, y_metric)
        label_points(ax, mine)
        note = f"{OURS_NOTE}, {PAPER_NOTE}" if theirs else f"{OURS_NOTE}; no paper values for this axis"
        ax.set(xlabel=x_label, ylabel=y_label, title=f"{title}\n{note}")
        model_legend(ax, mine, palette)
        written += save(fig, out, f"{name}_comparison{suffix}")
    return written


def draw_baselines(ax, claims: Claims, x_metric: str, y_metric: str) -> None:
    bx, by = baseline(claims, x_metric), baseline(claims, y_metric)
    if bx is not None:
        ax.axvline(bx, color=MUTED, ls="--", lw=1, zorder=1)
        ax.annotate(
            "majority answer",
            (bx, 0),
            xycoords=("data", "axes fraction"),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    if by is not None:
        ax.axhline(by, color=MUTED, ls="--", lw=1, zorder=1)
        ax.annotate(
            "majority answer",
            (0, by),
            xycoords=("axes fraction", "data"),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )


def bar_values(ax, xs, values: list[float | None], tops: list[float]) -> None:
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
                fontsize=LABEL_FONTSIZE,
                color=INK,
            )


def tick_name(name: str) -> str:
    """Model name on two lines, e.g. "Claude Opus 4.1\n(4.5)", so Fig 5a's ticks stay horizontal."""
    return name.replace(" (", "\n(")


def fig5a(bars: list[Bar], stem: str, out: Path, comparison: bool) -> list[Path]:
    with plt.rc_context(STYLE):
        width = max(FIG5A_MIN_WIDTH, FIG5A_INCHES_PER_MODEL * len(bars) + FIG5A_EXTRA_WIDTH)
        fig, ax = plt.subplots(figsize=(width, FIG5A_HEIGHT))
        xs = list(range(len(bars)))
        for series, (offset, kind, color) in enumerate(
            ((-BAR_WIDTH / 2, "text", TEXT_COLOR), (BAR_WIDTH / 2, "video", VIDEO_COLOR))
        ):
            pos = [x + offset for x in xs]
            values = [getattr(b, kind) or 0.0 for b in bars]
            cis = [getattr(b, f"{kind}_ci") for b in bars]
            err = (
                [
                    [v - c[0] if c else 0 for v, c in zip(values, cis, strict=True)],
                    [c[1] - v if c else 0 for v, c in zip(values, cis, strict=True)],
                ]
                if comparison
                else None
            )
            ax.bar(
                pos,
                values,
                BAR_WIDTH,
                color=color,
                edgecolor="white",
                linewidth=2,
                label=kind.capitalize(),
                yerr=err,
                error_kw={"elinewidth": 1, "ecolor": INK, "alpha": 0.6},
            )
            tops = [c[1] if comparison and c else v for v, c in zip(values, cis, strict=True)]
            bar_values(ax, pos, [getattr(b, kind) for b in bars], tops)
            if comparison:
                theirs = [(p, getattr(b, f"paper_{kind}")) for p, b in zip(pos, bars, strict=True)]
                ax.scatter(
                    [p for p, v in theirs if v is not None],
                    [v for _, v in theirs if v is not None],
                    marker="D",
                    s=PAPER_MARKER_SIZE,
                    facecolors="white",
                    edgecolors=INK,
                    zorder=3,
                    label="Paper" if series == 0 else None,  # one legend entry for both series
                )
        ax.set_xticks(xs, [tick_name(b.name) for b in bars], fontsize=TICK_FONTSIZE)
        ax.set(xlabel="Model", ylabel="Risk Recognition Accuracy", ylim=BAR_YLIM)
        if comparison:
            ax.set_title(f"{OURS_NOTE}, diamonds = paper")
        if bars:
            ax.legend(title="Accuracy Type", **OUTSIDE_LEGEND)
        return save(fig, out, stem)


def fig5b(run: RunData, claims: Claims, palette: dict[str, str], out: Path) -> list[Path]:
    x_label = "Last Time of Intervention Mean Absolute Error" + LOWER_INVERTED
    y_label, title = "Ratio of Accurate Interventions" + HIGHER, "Triggering Interventions"
    mine = ours(run, "timing_mae_s", "within_0_5_rate", y_scale=100)
    theirs = paper(run, claims, "5b", "timing_mae_s", "within_0_5_rate", y_scale=100)
    written = []
    with plt.rc_context(STYLE):
        for version, show_paper in (("paper", False), ("comparison", True)):
            fig, ax = plt.subplots(figsize=SCATTER_SIZE)
            scatter(ax, mine, palette, with_ci=show_paper)
            if show_paper:
                scatter(ax, theirs, palette, hollow=True)
            ax.invert_xaxis()
            label_points(ax, mine)
            ax.set(xlabel=x_label, ylabel=y_label, title=title + (f"\n{OURS_NOTE}, {PAPER_NOTE}" if show_paper else ""))
            model_legend(ax, mine, palette)
            written += save(fig, out, f"fig5b_{version}")
    return written


def headline_runs(results: Path, runs: list[RunData]) -> dict[str, RunData | None]:
    figures = (load_yaml(results / "headline.yaml", {}) or default_headline(runs)).get("figures") or {}
    by_name = {r.name: r for r in runs}
    return {fig: by_name.get((figures.get(fig) or {}).get("run") or "") for fig in ("fig4", "fig5", "fig6")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--severity", choices=sorted(SEVERITY), default="4way")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--claims", type=Path, default=CLAIMS_PATH)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "repro_fig4.yaml", help="model order = colors")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args(argv)

    runs = load_results(args.results)
    claims = load_claims(args.claims)
    palette = colors([m["name"] for m in yaml.safe_load(args.config.read_text())["models"]])
    chosen = headline_runs(args.results, runs)
    written: list[Path] = []
    if chosen["fig4"]:
        for name in FIG4:
            written += fig4(name, chosen["fig4"], claims, palette, args.severity, args.out)
    if chosen["fig4"] and chosen["fig5"]:
        written += fig5a(
            text_vs_video(chosen["fig4"], chosen["fig5"], claims, only_paper_models=True), "fig5a_paper", args.out, False
        )
        written += fig5a(
            text_vs_video(chosen["fig4"], chosen["fig5"], claims, only_paper_models=False), "fig5a_comparison", args.out, True
        )
    if chosen["fig5"]:
        written += fig5b(chosen["fig5"], claims, palette, args.out)
    print("fig6: not drawn yet, waiting for the image scorer (Riddhi)")
    missing = [f for f in ("fig4", "fig5") if not chosen[f]]
    if missing:
        print(f"no headline run for {', '.join(missing)} (results/headline.yaml)")
    print(f"plot: {len(written)} files under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
