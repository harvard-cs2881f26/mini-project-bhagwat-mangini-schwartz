#!/usr/bin/env python
"""Plot collected Figure 6a measurements; never substitute missing model results.

    python scripts/plot_constraints.py configs/repro_fig6_successors.yaml

Optional --categories CSV supplies a complete item mapping. Proposed annotations
require --allow-proposed-categories and are labeled in every category artifact.
Exports both within-category rates and shares of all violations.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "asimov-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asimov.data.rows import DATA_DIR, read_rows


# Appendix D, Table 5. Counts validate reconstruction; they never determine labels.
PAPER_CATEGORY_COUNTS = {
    "Gripper Geometry and Type": 37, "Material Properties": 31,
    "Commonsense Physicality": 26, "Logical Composition": 20,
    "Thermal": 18, "Safety and Special Conditions": 17,
    "Kinematics and Reach": 10, "Multi-arm and Coordination": 5,
}


def load_categories(path: Path, expected_ids: set[str], allow_proposed: bool = False):
    with path.open() as f:
        annotations = list(csv.DictReader(f))
    mapping = {}
    for row in annotations:
        item_id, category = row["id"], row["category"].strip()
        if item_id in mapping:
            raise ValueError(f"Duplicate category annotation: {item_id}")
        if category not in PAPER_CATEGORY_COUNTS:
            raise ValueError(f"Unknown category for {item_id}: {category!r}")
        mapping[item_id] = row
    if set(mapping) != expected_ids:
        raise ValueError("Category IDs must match the complete dataset exactly")
    proposed = any(row.get("review_status") != "reviewed" for row in annotations)
    if proposed and not allow_proposed:
        raise ValueError("Unreviewed mapping: pass --allow-proposed-categories for provisional outputs")
    return mapping, "proposed_unreviewed" if proposed else "reviewed_reconstruction"


def category_rates(summary: dict, mapping: dict, status: str):
    rows = []
    for category in PAPER_CATEGORY_COUNTS:
        ids = [i for i, r in mapping.items() if r["category"].strip() == category]
        outcomes = [summary["per_item"][i] for i in ids if i in summary["per_item"]]
        parsed = sum(o["violation"] is not None for o in outcomes)
        violations = sum(o["violation"] is True for o in outcomes)
        rows.append({"model": summary["model"], "category": category,
                     "mapping_status": status, "n_dataset_category": len(ids),
                     "n": len(outcomes), "n_missing": len(ids) - len(outcomes),
                     "n_parsed": parsed, "n_parse_failures": len(outcomes) - parsed,
                     "n_violations": violations, "n_model_violations": summary["n_violations"],
                     "n_ambiguous_assignments": sum(mapping[i].get("ambiguous", "").lower() == "true" for i in ids),
                     "within_category_rate": violations / parsed if parsed else None,
                     "share_of_violations": violations / summary["n_violations"] if summary["n_violations"] else None})
    return rows


def plot_category_panel(root: Path, summary: dict, rows: list[dict], status: str):
    # Figures 6b/c show Opus and GPT-5; do not invent equivalent panels for other models.
    model = summary["model"]
    panel = "fig6c" if model == "gpt-5" else "fig6b" if model.startswith("claude-opus-") else None
    if panel is None:
        return
    status_label = "PROVISIONAL: unreviewed reconstructed categories" if status == "proposed_unreviewed" else "Reviewed reconstructed categories; authors item labels unavailable"
    for metric, xlabel in [("share_of_violations", "Share of all parsed violations (%)"),
                           ("within_category_rate", "Violations / parsed responses within category (%)")]:
        ordered = sorted(rows, key=lambda r: r[metric] if r[metric] is not None else -1)
        fig, ax = plt.subplots(figsize=(11, 6))
        values = [100 * r[metric] if r[metric] is not None else 0 for r in ordered]
        labels = [r["category"] for r in ordered]
        bars = ax.barh(labels, values, color="#426b9e")
        ax.bar_label(bars, labels=[f"{v:.1f}% ({r['n_violations']}/{r['n_model_violations'] if metric == 'share_of_violations' else r['n_parsed']})" if r[metric] is not None else "N/A" for v, r in zip(values, ordered)], padding=4, fontsize=9)
        ax.set_xlim(0, max(values + [1]) * 1.4)
        ax.set_xlabel(xlabel)
        substitute = " (substitute for Opus 4.1)" if panel == "fig6b" and model != "claude-opus-4-1-20250805" else ""
        ax.set_title(f"Figure {panel[-2:]}-style category breakdown — {model}{substitute}\n{status_label}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        note = "Proposed labels use constraint text only; category totals are not forced to match the paper."
        note += "\nParse failures excluded. Distribution denominator is explicit; authors denominator is not verified."
        fig.text(.02, .02, note, fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, 1))
        suffix = "" if metric == "share_of_violations" else "_within_category"
        for ext in ("png", "pdf"):
            fig.savefig(root / f"{panel}{suffix}.{ext}", dpi=180)
        plt.close(fig)


def plot_model_rates(root: Path, summaries: list[dict], unavailable=(), extra_note: str = ""):
    """Figure 6a format: horizontal bars, highest rate first, viridis colors."""
    summaries = sorted(summaries, key=lambda row: row["violation_rate"], reverse=True)
    root.mkdir(parents=True, exist_ok=True)
    fields = ["name", "model", "run", "thinking", "message_roles", "n_dataset", "n_requested",
              "n", "n_parsed", "n_missing", "n_violations", "violation_rate",
              "parse_failure_rate", "n_empty", "violation_rate_lower_bound", "violation_rate_upper_bound"]
    with (root / "fig6a.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=(16, max(6, .85 * len(summaries) + 2)))
        values = [100 * row["violation_rate"] for row in summaries]
        labels = [row.get("name", row["model"]) for row in summaries]
        cmap = plt.get_cmap("viridis")
        colors = [cmap(.88 - .75 * i / max(len(values) - 1, 1)) for i in range(len(values))]
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=.82)
        partial = [row["n"] < row.get("n_dataset", 164) for row in summaries]
        for bar, incomplete in zip(bars, partial):
            if incomplete:
                bar.set_hatch("//")
                bar.set_edgecolor("#333333")
        ax.bar_label(bars, labels=[f"{v:.1f}" + (" (partial)" if part else "")
                                 for v, part in zip(values, partial)], padding=7, fontsize=19)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 10))
        ax.tick_params(axis="both", labelsize=13)
        ax.set_xlabel("Constraint Violation Rate (%)", fontsize=19)
        ax.set_ylabel("Model / configuration", fontsize=17)
        ax.set_title("Model Constraint Violation Rate (Highest to Lowest)", fontsize=22, pad=22)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="white", linewidth=1.2)
        ax.set_axisbelow(True)
        note = "164 examples per full run; rates exclude malformed responses. Lower is better. Settings and counts are in the CSV."
        if any(partial):
            note += "\nHatched bars have partial coverage and are not full-benchmark rates."
        if unavailable:
            note += "\nNo scored result: " + ", ".join(unavailable)
        if extra_note:
            note += "\n" + extra_note
        fig.text(.02, .025, note, fontsize=10, linespacing=1.5)
        fig.tight_layout(rect=(0, .12, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(root / f"fig6a.{ext}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--categories", type=Path)
    parser.add_argument("--allow-proposed-categories", action="store_true",
                        help="Generate explicitly provisional outputs from unreviewed labels")
    parser.add_argument("--write-category-template", action="store_true",
                        help="Write a blank review CSV in the gitignored data directory")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    variant = cfg.get("variant", cfg.get("benchmark", "image"))
    if args.write_category_template:
        data_root = DATA_DIR / variant
        target = data_root / "category_review.csv"
        if target.exists():
            print(f"Preserved existing review file: {target}")
            return
        with target.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "category", "review_status", "constraint"], lineterminator="\n")
            writer.writeheader()
            for row in read_rows(variant):
                constraint = row["prompt"].partition("<CONSTRAINTS START>")[2].partition("<CONSTRAINTS END>")[0]
                writer.writerow({"id": row["id"], "category": "", "review_status": "proposed", "constraint": constraint})
        print(f"Wrote {target}; category labels need human review.")
        return
    root = ROOT / "results" / cfg["run"]
    summaries = []
    unavailable = []
    for spec in cfg["models"]:
        path = root / spec["model"] / "summary.json"
        if not path.exists():
            unavailable.append(spec["model"])
            continue
        row = json.loads(path.read_text())
        if row.get("violation_rate") is None:
            unavailable.append(spec["model"])
            continue
        row["run"] = cfg["run"]
        row.setdefault("message_roles", cfg.get("message_roles", "user"))
        row["name"] += f" ({row['thinking']}; {row['message_roles']})"
        summaries.append(row)
    if not summaries:
        raise SystemExit("No scored responses available; cannot plot real measurements.")
    plot_model_rates(root, summaries, unavailable)
    if args.categories:
        expected_ids = {row["id"] for row in read_rows(variant)}
        mapping, status = load_categories(args.categories, expected_ids, args.allow_proposed_categories)
        rows = []
        for summary in summaries:
            model_rows = category_rates(summary, mapping, status)
            rows.extend(model_rows)
            plot_category_panel(root, summary, model_rows, status)
        with (root / "category_rates.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with (root / "category_counts.csv").open("w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["category", "reconstructed_count", "paper_count", "difference", "mapping_status"])
            for category, paper_count in PAPER_CATEGORY_COUNTS.items():
                count = sum(r["category"].strip() == category for r in mapping.values())
                writer.writerow([category, count, paper_count, count - paper_count, status])
        import hashlib
        provenance = {"mapping_status": status, "mapping_sha256": hashlib.sha256(args.categories.read_bytes()).hexdigest(),
                      "mapping_local_path": str(args.categories), "mapping_basis": "constraint text only; outcomes not used to assign labels",
                      "paper": "https://arxiv.org/html/2509.21651v2#A4", "authors_item_labels_available": False,
                      "denominators": {"share_of_violations": "category violations / all parsed model violations; caption-based reconstruction, not verified authors formula",
                                       "within_category_rate": "category violations / parsed category responses"},
                      "responses_sha256": {s["model"]: hashlib.sha256((root / s["model"] / "responses.jsonl").read_bytes()).hexdigest() for s in summaries}}
        (root / "category_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {root / 'fig6a.png'}, PDF and CSV")


if __name__ == "__main__":
    main()
