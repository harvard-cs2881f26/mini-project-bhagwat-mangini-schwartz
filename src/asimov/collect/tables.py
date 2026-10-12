"""results/index.csv (one row per run x model) and results/per_item.csv (run x model x item x question)."""

from __future__ import annotations

import csv
import io

import numpy as np

from asimov.claims import is_stand_in
from asimov.collect.load import ModelResult, RunData
from asimov.collect.metrics import item_rows

LEADING = (
    "run",
    "benchmark",
    "name",
    "provider",
    "model",
    "served_model",
    "stand_in",
    "prompt",
    "thinking",
    "log_reasoning",
    "key",
    "n",
    "errors",
    "config_path",
    "config_sha256",
    "git_commit",
    "git_dirty",
    "scored_at",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "latency_p50_s",
    "latency_p95_s",
    "cost_usd",
)
PER_ITEM_COLUMNS = ("run", "benchmark", "model", "item", "question", "pred", "gold", "correct", "status", "parse_ok", "finish")
PER_MTOK = 1e6
LATENCY_DIGITS = 3  # rounded so index.csv is the same on every machine
COST_DIGITS = 6


def cost_usd(summary: dict, prices: dict) -> float | None:
    """$ from logged tokens x configs/prices.yaml (per million tokens); None when the model has no price."""
    price = prices.get(summary.get("model"))
    if not price:
        return None
    cached = summary.get("cached_input_tokens") or 0
    fresh = (summary.get("input_tokens") or 0) - cached
    cached_rate = price.get("cached_input_per_mtok", price["input_per_mtok"])
    total = (
        fresh * price["input_per_mtok"] + cached * cached_rate + (summary.get("output_tokens") or 0) * price["output_per_mtok"]
    )
    return round(total / PER_MTOK, COST_DIGITS)


def index_row(run: RunData, model: ModelResult, prices: dict) -> dict:
    s = model.summary
    lat = np.asarray(model.latencies, dtype=float)
    p50, p95 = (round(float(np.percentile(lat, q)), LATENCY_DIGITS) for q in (50, 95)) if len(lat) else (None, None)
    made = {
        "run": run.name,
        "benchmark": run.benchmark,
        "served_model": ";".join(model.served_models),
        "stand_in": is_stand_in(s["name"]),
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "cost_usd": cost_usd(s, prices),
    }
    return {**s, **made}


def index_rows(runs: list[RunData], prices: dict) -> list[dict]:
    return [index_row(run, m, prices) for run in runs for m in run.models]


def per_item_rows(runs: list[RunData]) -> list[dict]:
    rows = []
    for run in runs:
        for m in run.models:
            for item_id in sorted(m.per_item):
                outcome = m.per_item[item_id]
                for r in item_rows(run.benchmark, outcome):
                    rows.append(
                        {
                            "run": run.name,
                            "benchmark": run.benchmark,
                            "model": m.summary["model"],
                            "item": item_id,
                            **r,
                            "parse_ok": outcome.get("parse_ok"),
                            "finish": outcome.get("finish"),
                        }
                    )
    return rows


def cell(v) -> str:
    return "" if v is None else str(v)


def to_csv(rows: list[dict], leading: tuple[str, ...]) -> str:
    """CSV with `leading` columns first, then every other key sorted; missing values are empty."""
    extra = sorted({k for r in rows for k in r} - set(leading))
    columns = [c for c in leading if any(c in r for r in rows) or not rows] + extra
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for r in rows:
        writer.writerow([cell(r.get(c)) for c in columns])
    return out.getvalue()
