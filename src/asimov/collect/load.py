"""Read results/ into memory: the only I/O in asimov.collect (the script writes the pages).

Reads each run folder's (results/<id>_<run>/) `config.yaml`, `<model>/summary.json` and `responses.jsonl`
(latency, served model id, reply text for comparison excerpts), and the run's hand-written `writeup.md` if any.
A run's name comes from its config.yaml `run`, else its summaries' `run`: never from the folder name.
Never reads data/: scorers save pred, gold and correct per item in summary.json (docs/PLAN.md §11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from asimov.run_ids import UNREGISTERED_PROVIDERS

SKIP_PROVIDERS = UNREGISTERED_PROVIDERS  # free smoke runs are gitignored, so they never appear on pages
WRITEUP_FILE = "writeup.md"  # results/<id>_<run>/writeup.md: the owner's write-up, shown on the run page


@dataclass(frozen=True)
class ModelResult:
    summary: dict  # summary.json without per_item
    per_item: dict[str, dict]
    latencies: tuple[float, ...]  # replies under the summary's key, for the items it scored (rows with latency_s)
    served_models: tuple[str, ...]
    replies: dict[str, str]  # item id -> reply text
    usage: dict[str, dict] = field(default_factory=dict)  # item id -> the reply's token usage (incl. Gemini thinking_tokens)
    probs: dict[str, dict] = field(default_factory=dict)  # item id -> question -> letter -> probability (decision models)
    item_latency: dict[str, float] = field(default_factory=dict)  # item id -> latency_s, for replies that have one

    @property
    def name(self) -> str:
        return self.summary["name"]


@dataclass(frozen=True)
class RunData:
    name: str  # the logical run name (config `run`), e.g. repro_fig4
    folder: str  # the run's folder under results/, e.g. txt0_repro_fig4; pages link to it
    benchmark: str
    config: dict | None  # <folder>/config.yaml, written by the runner at run start
    models: tuple[ModelResult, ...]  # in the config's order
    writeup: str | None = None  # <folder>/writeup.md, hand-written; read, never written


def load_model(mdir: Path) -> ModelResult:
    full = json.loads((mdir / "summary.json").read_text())
    per_item = full.pop("per_item", {}) or {}
    key = full.get("key")
    latencies, served, replies, usage, probs, item_latency = [], set(), {}, {}, {}, {}
    path = mdir / "responses.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row.get("key") != key or row.get("id") not in per_item:
            continue
        if row.get("latency_s") is not None:  # a row without timing is left out of the latency stats, not counted as 0
            latencies.append(float(row["latency_s"]))
            item_latency[row["id"]] = float(row["latency_s"])
        served.add(row.get("served_model") or "")
        replies[row["id"]] = row.get("text", "")
        usage[row["id"]] = row.get("usage") or {}
        if row.get("probs"):
            probs[row["id"]] = row["probs"]
    return ModelResult(full, per_item, tuple(latencies), tuple(sorted(served - {""})), replies, usage, probs, item_latency)


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
    names = {config["run"]} if config and config.get("run") else {r.summary.get("run") for r in results}
    if len(names) != 1 or None in names:
        found = sorted(map(str, names))
        raise ValueError(f"{run_dir}: no run name in config.yaml, and the summaries lack or disagree on `run` ({found})")
    writeup_path = run_dir / WRITEUP_FILE
    writeup = writeup_path.read_text() if writeup_path.exists() else None
    ordered = tuple(model_order(config, results))
    return RunData(str(names.pop()), run_dir.name, str(benchmarks.pop()), config, ordered, writeup)


def load_results(results_dir: Path) -> list[RunData]:
    """Every run under results/ that has at least one non-mock model summary, by run name."""
    runs = [r for r in (load_run(d) for d in sorted(results_dir.iterdir()) if d.is_dir()) if r is not None]
    seen: dict[str, str] = {}
    for r in runs:
        if r.name in seen:
            raise ValueError(f"run {r.name!r} is in two folders: results/{seen[r.name]}/ and results/{r.folder}/")
        seen[r.name] = r.folder
    return sorted(runs, key=lambda r: r.name)


def load_yaml(path: Path, default):
    return yaml.safe_load(path.read_text()) if path.exists() else default
