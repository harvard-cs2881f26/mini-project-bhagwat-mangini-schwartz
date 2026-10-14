"""Run IDs (txt0, vid0, ...) and each run's folder, results/<id>_<run>/.

results/run_ids.yaml maps run name -> ID. An ID, once given, is never changed or reused, even after its run is
deleted, so a run ID quoted in a slide or a note always means the same run. The runner gives a new run its ID
when it first writes results (`run_dir`); scripts/collect.py adds any run still missing one. Descriptions and
the scoring-order assignment for collect are in asimov.collect.run_ids.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

import yaml

PREFIXES = {"text": "txt", "video": "vid", "image": "img", "image_rationale": "imgr"}  # benchmark -> ID prefix
RUN_IDS_FILE = "run_ids.yaml"
RUN_IDS_HEADER = """\
# run name -> run ID. Maintained by the runner and scripts/collect.py, which add every new run: don't edit by hand.
# An ID is the benchmark prefix plus a number; it is never changed or reused, even after a run is deleted.
# Each run's folder is results/<id>_<run>/.
"""
ID_PATTERN = re.compile(r"^([a-z]+)(\d+)$")


def prefix(benchmark: str) -> str:
    if benchmark not in PREFIXES:
        raise ValueError(f"no run ID prefix for benchmark {benchmark!r}; add it to PREFIXES in asimov/run_ids.py")
    return PREFIXES[benchmark]


def next_number(ids: dict[str, str], run_prefix: str) -> int:
    """One more than the highest number any ID (past or present) has under this exact prefix."""
    used = [int(m[2]) for v in ids.values() if (m := ID_PATTERN.match(v)) and m[1] == run_prefix]
    return max(used, default=-1) + 1


def parse_ids(raw: dict | None) -> dict[str, str]:
    """results/run_ids.yaml as loaded; fails on any entry that isn't `run name: id` (e.g. the old {id, description})."""
    ids = {}
    for name, value in (raw or {}).items():
        if not isinstance(value, str) or not ID_PATTERN.match(value):
            raise ValueError(f"results/{RUN_IDS_FILE}: entry {name!r} must be run name -> id (e.g. txt0), got {value!r}")
        ids[str(name)] = value
    return ids


def run_ids_yaml(ids: dict[str, str]) -> str:
    return RUN_IDS_HEADER + yaml.safe_dump(ids, sort_keys=False)


def read_ids(results_dir: Path) -> dict[str, str]:
    path = results_dir / RUN_IDS_FILE
    return parse_ids(yaml.safe_load(path.read_text()) if path.exists() else None)


def folder_name(run_id: str, run_name: str) -> str:
    return f"{run_id}_{run_name}"


UNREGISTERED_PROVIDERS = frozenset({"mock", "mock_decision"})  # free smoke runs: no ID, plain results/<run>/ (gitignored)


def run_dir(results_dir: Path, run_name: str, benchmark: str, providers: Iterable[str] = ()) -> Path:
    """results/<id>_<run>/ for this run. A run without an ID gets the next number for its benchmark's prefix,
    written to results/run_ids.yaml. A legacy results/<run>/ folder is renamed to the new name. Creates nothing else.
    A run whose models are all mock providers gets no ID and keeps results/<run>/, so smoke runs never use up a number.
    """
    providers = set(providers)
    if providers and providers <= UNREGISTERED_PROVIDERS:
        return results_dir / run_name
    ids = read_ids(results_dir)
    if run_name not in ids:
        p = prefix(benchmark)
        ids[run_name] = f"{p}{next_number(ids, p)}"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / RUN_IDS_FILE).write_text(run_ids_yaml(ids))
    path = results_dir / folder_name(ids[run_name], run_name)
    legacy = results_dir / run_name
    if legacy.is_dir() and not path.exists():
        os.rename(legacy, path)
        print(f"Renamed {legacy} -> {path} (run folders are now results/<id>_<run>/)")
    return path
