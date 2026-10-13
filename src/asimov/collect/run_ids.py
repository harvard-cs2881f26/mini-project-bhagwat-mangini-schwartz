"""Short, stable run IDs (txt0, vid0, ...) and owners' run descriptions. Pure: the I/O is in load.py and collect.py.

Both live in results/run_ids.yaml (run name -> {id, description}). An ID, once given, is never changed or reused,
even after its run is deleted, so a run ID quoted in a slide or a note always means the same run. Descriptions
are hand-written by the run's owner; they stay out of configs, whose fingerprint decides reuse of saved replies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import yaml

from asimov.collect.load import RunData

PREFIXES = {"text": "txt", "video": "vid", "image": "img", "image_rationale": "imgr"}  # benchmark -> ID prefix
RUN_IDS_FILE = "run_ids.yaml"
RUN_IDS_HEADER = """\
# run name -> {id, description}. scripts/collect.py adds every new run here.
# id: automatic (benchmark prefix + number, in scoring order); never changed or reused, even after a run is deleted.
# description: hand-written by the run's owner (what the run tests, not its results). collect.py adds new runs
# with description "" (pages then show an automatic one from the settings) and never overwrites a description.
"""
ID_PATTERN = re.compile(r"^([a-z]+)(\d+)$")
NEVER_SCORED = datetime.max.replace(tzinfo=timezone.utc)  # a run without scored_at sorts last
AUTO_MARK = "(auto)"


@dataclass(frozen=True)
class RunLabel:
    id: str
    description: str = ""


def first_scored(run: RunData) -> datetime:
    """When the run's first model was scored (summary `scored_at`)."""
    stamps = [datetime.fromisoformat(s) for m in run.models if (s := m.summary.get("scored_at"))]
    return min(stamps, default=NEVER_SCORED)


def prefix(benchmark: str) -> str:
    if benchmark not in PREFIXES:
        raise ValueError(f"no run ID prefix for benchmark {benchmark!r}; add it to PREFIXES in asimov/collect/run_ids.py")
    return PREFIXES[benchmark]


def next_number(ids: dict[str, str], run_prefix: str) -> int:
    """One more than the highest number any ID (past or present) has under this exact prefix."""
    used = [int(m[2]) for v in ids.values() if (m := ID_PATTERN.match(v)) and m[1] == run_prefix]
    return max(used, default=-1) + 1


def assign_ids(existing: dict[str, str], runs: Iterable[RunData]) -> dict[str, str]:
    """Every existing entry unchanged; each new run gets the next free number for its prefix, by first scored_at, then name."""
    ids = dict(existing)
    for run in sorted((r for r in runs if r.name not in existing), key=lambda r: (first_scored(r), r.name)):
        p = prefix(run.benchmark)
        ids[run.name] = f"{p}{next_number(ids, p)}"
    return ids


def update_labels(existing: dict[str, RunLabel], runs: Iterable[RunData]) -> dict[str, RunLabel]:
    """Existing entries unchanged (id and description); new runs get an ID and an empty description."""
    ids = assign_ids({name: label.id for name, label in existing.items()}, runs)
    return {name: existing.get(name) or RunLabel(run_id) for name, run_id in ids.items()}


def missing_labels(existing: dict[str, RunLabel], runs: Iterable[RunData]) -> list[str]:
    return sorted(r.name for r in runs if r.name not in existing)


def parse_labels(raw: dict | None) -> dict[str, RunLabel]:
    """results/run_ids.yaml as loaded; fails on any entry that isn't {id: str, description: str}."""
    labels = {}
    for name, entry in (raw or {}).items():
        if not isinstance(entry, dict) or set(entry) != {"id", "description"} or not isinstance(entry["id"], str):
            raise ValueError(f"results/{RUN_IDS_FILE}: entry {name!r} must be {{id: <str>, description: <str>}}, got {entry!r}")
        labels[str(name)] = RunLabel(entry["id"], entry["description"] or "")
    return labels


def run_ids_yaml(labels: dict[str, RunLabel]) -> str:
    return RUN_IDS_HEADER + yaml.safe_dump({name: asdict(label) for name, label in labels.items()}, sort_keys=False)


def auto_description(run: RunData) -> str:
    """Fallback from the run's settings, e.g. "text · prompt text_verbatim · thinking min · 8 models (auto)"."""
    prompts = ", ".join(sorted({str(m.summary.get("prompt")) for m in run.models}))
    thinking = ", ".join(sorted({str(m.summary.get("thinking")) for m in run.models}))
    n = len(run.models)
    return f"{run.benchmark} · prompt {prompts} · thinking {thinking} · {n} model{'s' if n != 1 else ''} {AUTO_MARK}"


def description(run: RunData, labels: dict[str, RunLabel]) -> str:
    label = labels.get(run.name)
    return label.description if label and label.description else auto_description(run)
