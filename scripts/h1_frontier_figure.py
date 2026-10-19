#!/usr/bin/env python
"""H1 (text) frontier charts for the write-up: held-out classifier + LLM mixes against cost and median latency.

    python scripts/h1_frontier_figure.py   # -> figures/h1_frontier/accuracy_vs_cost.png, accuracy_vs_speed.png

Free: reads results/router_search/{frontier,candidates}.csv and results/index.csv only. EXPLORATORY: the blue
points are the router search's held-out picks (WORKLOG 2026-10-13); the single systems are each at their best
setting, picked on all 319 items, so they are optimistic. Median latency hides the slow half of a routed mix.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "h1_frontier"
Q4 = ("latent_risk_accuracy", "latent_severity_accuracy", "effect_accuracy", "activated_risk_accuracy")
BLUE, ORANGE, INK = "#1F4E9E", "#E2581E", "#16181A"

# (label, candidate name, label offset in points for the cost chart, for the speed chart)
LLMS = [
    ("GPT-5", "GPT-5 [min]", (0, -18), (0, -18)),
    ("Mini", "GPT-5 Mini [min]", (0, -18), (0, -18)),
    ("Flash-Lite", "Gemini 3.1 Flash-Lite [min]", (0, -18), (0, -18)),
]
CLASSIFIERS = [
    ("Jev", "jev-latest (part2_jev_full_cumulative)", (0, 10), (0, 10)),
    ("OpenJev", "OpenJev think 512 full", (0, 10), (0, 10)),
    ("luna", "gpt-6-luna (part2_luna_decide_full_cumulative)", (0, -18), (0, -18)),
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    singles = {
        r["candidate"]: (float(r["mean_of_4"]), float(r["cost_per_1000_usd"]), float(r["latency_p50_s"]))
        for r in read(ROOT / "results/router_search/candidates.csv")
        if r["family"] == "single"
    }
    held = [r for r in read(ROOT / "results/router_search/frontier.csv") if r["set"] == "held_out"]
    flash_lite_high = None
    for r in read(ROOT / "results/index.csv"):
        if r["run"] == "repro_fig4_high" and r["name"].startswith("Gemini 2.5 Flash-Lite"):
            acc = sum(float(r[k]) for k in Q4) / 4
            flash_lite_high = (acc, float(r["cost_usd"]) / 319 * 1000, float(r["latency_p50_s"]))
    gpt5 = singles["GPT-5 [min]"][0]
    OUT.mkdir(parents=True, exist_ok=True)

    for axis, idx, xlabel, name in (
        ("cost", 1, "$ per 1,000 items (log)", "accuracy_vs_cost"),
        ("latency", 2, "median latency per item, s (log)", "accuracy_vs_speed"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.set_xscale("log")
        ax.axhline(gpt5, color=INK, ls="--", lw=1.2)
        if axis == "latency":
            ax.axvline(2.5, color="#B4430F", ls="--", lw=1)
            ax.text(2.4, 0.815, "2.5 s to act", color="#B4430F", ha="right", fontsize=9)
        for label, cand, off_cost, off_lat in LLMS:
            a = singles[cand]
            ax.plot(a[idx], a[0], "s", color=INK, ms=8)
            ax.annotate(
                label,
                (a[idx], a[0]),
                textcoords="offset points",
                xytext=off_cost if axis == "cost" else off_lat,
                ha="center",
                fontsize=9,
            )
        for label, cand, off_cost, off_lat in CLASSIFIERS:
            a = singles[cand]
            x = max(a[idx], 0.04) if axis == "cost" else a[idx]
            free = axis == "cost" and a[idx] == 0
            ax.plot(x, a[0], "^", color="white" if free else ORANGE, mec="#B4430F", ms=9, mew=1.5)
            ax.annotate(
                label + (" (free tier)" if free else ""),
                (x, a[0]),
                textcoords="offset points",
                xytext=off_cost if axis == "cost" else off_lat,
                ha="center",
                fontsize=9,
                color="#B4430F",
            )
        if flash_lite_high:
            a = flash_lite_high
            ax.plot(a[idx], a[0], "s", color="white", mec=INK, ms=8, mew=1.5)
            offset = (0, -18) if axis == "cost" else (0, 10)
            ax.annotate("Flash-Lite high", (a[idx], a[0]), textcoords="offset points", xytext=offset, ha="center", fontsize=9)
        pts = [r for r in held if r["axis"] == axis]
        xs = [float(r["x"]) for r in pts]
        ys = [float(r["mean_of_4"]) for r in pts]
        lo = [y - float(r["lo"]) for y, r in zip(ys, pts, strict=True)]
        hi = [float(r["hi"]) - y for y, r in zip(ys, pts, strict=True)]
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            color=BLUE,
            marker="o",
            ms=7,
            lw=2,
            capsize=3,
            label="classifier + LLM mix, held-out (95% range)",
        )
        ax.set_ylim(0.60, 0.82)
        ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80])
        ax.set_yticklabels([f"{t}%" for t in (60, 65, 70, 75, 80)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy, mean of 4 questions")
        ax.grid(alpha=0.25)
        ax.plot([], [], "^", color=ORANGE, mec="#B4430F", label="classifier alone (best setting)")
        ax.plot([], [], "s", color=INK, label="LLM alone (thinking min)")
        ax.legend(loc="lower right", fontsize=8)
        title = "cost" if axis == "cost" else "speed"
        ax.set_title(f"EXPLORATORY. H1 (text): accuracy vs {title}, n = 319", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=160)
        plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
