#!/usr/bin/env python
"""Offline injury/constraint slices of canonical scored runs; no training or API calls.

    python scripts/category_effects.py results/txt0_repro_fig4/*/summary.json
    python scripts/category_effects.py results/img3_repro_fig6_default/*/summary.json \
        --categories data/image/category_proposals.csv --allow-proposed-categories

Writes CSV to stdout (or --output). Text categories use each item's own gold injury
option TEXT, never its varying letter. Video retains the released injury codes;
No_Consensus is its own group. Image reports violation rate, not accuracy. Labels
are joined after inference and never sent to models. CIs describe item sampling,
not repeat-run variability or causal effects of category. Compare run IDs, not
just thinking labels: some candidate configs override native reasoning settings.
No significance claim
is made; category, severity, scene difficulty and model can be confounded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from asimov.bench.text_score import QUESTIONS  # noqa: E402
from asimov.data.rows import DATA_DIR  # noqa: E402
from asimov.data.text import load_text_answers, load_text_options  # noqa: E402
from asimov.data.video import load_video_answers  # noqa: E402
from plot_constraints import load_categories  # noqa: E402


def wilson(hits: int, n: int) -> tuple[float | None, float | None]:
    """95% Wilson interval; undefined for an empty denominator."""
    if not n:
        return None, None
    z = 1.959963984540054
    p = hits / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def category_labels(benchmark: str, data_dir: Path, categories: Path | None, allow_proposed: bool) -> tuple[dict[str, str], str]:
    if benchmark == "text":
        answers = load_text_answers(data_dir)
        options = load_text_options(data_dir)
        return {
            item_id: options[item_id]["latent_risk"][answer["latent_risk"]].strip() for item_id, answer in answers.items()
        }, "released_gold_option_text"
    if benchmark == "video":
        return {
            item_id: answer.injury_type for item_id, answer in load_video_answers(data_dir).items()
        }, "released_gold_injury_code"
    if benchmark != "image":
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if categories is None:
        raise ValueError("Image requires --categories; authors' per-item taxonomy is not released")
    ids = {json.loads(line)["id"] for line in (data_dir / "image" / "items.jsonl").read_text().splitlines() if line.strip()}
    mapping, status = load_categories(categories, ids, allow_proposed)
    return {item_id: row["category"].strip() for item_id, row in mapping.items()}, status


def slices(summary: dict, labels: dict[str, str], status: str) -> list[dict]:
    """Use existing per-item decisions unchanged; expose unknown/missing separately."""
    benchmark = summary["benchmark"]
    outcomes = summary["per_item"]
    unknown_ids = set(outcomes) - set(labels)
    if unknown_ids:
        raise ValueError(f"{len(unknown_ids)} scored items have no category label")
    metrics = (
        QUESTIONS
        if benchmark == "text"
        else (("risk_accuracy", "within_0_5_rate") if benchmark == "video" else ("violation_rate",))
    )
    rows = []
    for category in sorted(set(labels.values())):
        ids = [item_id for item_id, label in labels.items() if label == category]
        returned = [outcomes[item_id] for item_id in ids if item_id in outcomes]
        for metric in metrics:
            if benchmark == "text":
                values = [bool(item[metric]["correct"]) for item in returned]
                unread = sum(item[metric]["pred"] is None for item in returned)
            elif benchmark == "video":
                key = "risk_ok" if metric == "risk_accuracy" else "within_0_5"
                values = [bool(item[key]) for item in returned]
                # Timing cases include parsing status; risk_ok alone cannot distinguish
                # an incorrect answer from an unreadable answer. Do not invent counts.
                unread = None
            else:
                values = [item["violation"] for item in returned if item["violation"] is not None]
                unread = len(returned) - len(values)
            hits, denominator = sum(values), len(values)
            low, high = wilson(hits, denominator)
            missing = len(ids) - len(returned)
            unknown = len(ids) - denominator
            rows.append(
                {
                    "run": summary.get("run"),
                    "model": summary["model"],
                    "thinking": summary.get("thinking"),
                    "message_roles": summary.get("message_roles"),
                    "config_path": summary.get("config_path"),
                    "config_sha256": summary.get("config_sha256"),
                    "benchmark": benchmark,
                    "category": category,
                    "mapping_status": status,
                    "metric": metric,
                    "n_dataset_category": len(ids),
                    "n_returned": len(returned),
                    "n_missing": missing,
                    "n_unread": unread,
                    "numerator": hits,
                    "denominator": denominator,
                    "rate": hits / denominator if denominator else None,
                    "ci95_low": low,
                    "ci95_high": high,
                    "all_items_lower_bound": hits / len(ids),
                    "all_items_upper_bound": (hits + unknown) / len(ids),
                    "run_complete": set(outcomes) == set(labels),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--categories", type=Path)
    parser.add_argument("--allow-proposed-categories", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.summaries:
        summary = json.loads(path.read_text())
        if "per_item" not in summary:
            parser.error(f"Pass model-level summary.json, not combined summary: {path}")
        labels, status = category_labels(summary["benchmark"], args.data_dir, args.categories, args.allow_proposed_categories)
        rows.extend(slices(summary, labels, status))
    if not rows:
        parser.error("No category rows")
    if args.output:
        with args.output.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} category measurements to {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
