"""Read results/ into memory: the only I/O in asimov.collect (the script writes the pages).

Reads `results/<run>/config.yaml`, `<run>/<model>/summary.json` and `responses.jsonl` (latency,
served model id, reply text for comparison excerpts). Never reads data/: scorers save pred, gold
and correct per item in summary.json (docs/PLAN.md §11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

SKIP_PROVIDERS = frozenset({"mock"})  # free smoke runs are gitignored, so they never appear on pages


@dataclass(frozen=True)
class ModelResult:
    summary: dict  # summary.json without per_item
    per_item: dict[str, dict]
    latencies: tuple[float, ...]  # replies under the summary's key, for the items it scored (rows with latency_s)
    served_models: tuple[str, ...]
    replies: dict[str, str]  # item id -> reply text

    @property
    def name(self) -> str:
        return self.summary["name"]


@dataclass(frozen=True)
class RunData:
    name: str
    benchmark: str
    config: dict | None  # results/<run>/config.yaml, written by the runner at run start
    models: tuple[ModelResult, ...]  # in the config's order


def load_model(mdir: Path) -> ModelResult:
    full = json.loads((mdir / "summary.json").read_text())
    per_item = full.pop("per_item", {}) or {}
    key = full.get("key")
    latencies, served, replies = [], set(), {}
    path = mdir / "responses.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row.get("key") != key or row.get("id") not in per_item:
            continue
        if row.get("latency_s") is not None:  # a row without timing is left out of the latency stats, not counted as 0
            latencies.append(float(row["latency_s"]))
        served.add(row.get("served_model") or "")
        replies[row["id"]] = row.get("text", "")
    return ModelResult(full, per_item, tuple(latencies), tuple(sorted(served - {""})), replies)


def model_order(config: dict | None, results: list[ModelResult]) -> list[ModelResult]:
    """Config order (as in the run's tables), then any model not in the config, by name."""
    listed = [m["model"] for m in (config or {}).get("models", [])]
    rank = {model: i for i, model in enumerate(listed)}
    return sorted(results, key=lambda r: (rank.get(r.summary["model"], len(listed)), r.summary["model"]))


def load_run(run_dir: Path) -> RunData | None:
    results = [load_model(p.parent) for p in sorted(run_dir.glob("*/summary.json"))]
    results = [r for r in results if r.summary.get("provider") not in SKIP_PROVIDERS]
    if not results:
        return None
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else None
    benchmarks = {r.summary.get("benchmark") for r in results}
    if len(benchmarks) != 1 or None in benchmarks:
        raise ValueError(f"{run_dir}: model summaries lack or disagree on `benchmark` ({sorted(map(str, benchmarks))})")
    return RunData(run_dir.name, str(benchmarks.pop()), config, tuple(model_order(config, results)))


def load_results(results_dir: Path) -> list[RunData]:
    """Every run under results/ that has at least one non-mock model summary, by run name."""
    runs = (load_run(d) for d in sorted(results_dir.iterdir()) if d.is_dir())
    return [r for r in runs if r is not None]


def load_yaml(path: Path, default):
    return yaml.safe_load(path.read_text()) if path.exists() else default
