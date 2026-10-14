"""Every committed run records how it was made (CLAUDE.md: results only come from scripts/run.py).

For each run folder (results/<id>_<run>/) with a summary.json: the config copy exists and validates, every model's
summary.json has the provenance fields, and the saved replies behind it carry the fields we rely
on. Reads only results/, so it runs on CI without the dataset. Results made with an older runner,
or outside scripts/run.py, fail here: rescore them with `scripts/run.py <config> --no-calls` (free).
"""

import json
from pathlib import Path

import pytest
import yaml

from asimov.config import RunConfig
from asimov.run_ids import folder_name, read_ids

RESULTS = Path(__file__).resolve().parents[1] / "results"
SUMMARY_FIELDS = {
    "name",
    "model",
    "benchmark",
    "run",
    "key",
    "config_path",
    "config_sha256",
    "git_commit",
    "git_dirty",
    "scored_at",
    "n",
    "errors",
}
SCORED_FIELDS = {"provider", "prompt", "thinking", "input_tokens", "output_tokens", "truncation_rate", "refusal_rate", "per_item"}
REPLY_FIELDS = {
    "id",
    "key",
    "provider",
    "model",
    "served_model",
    "prompt",
    "thinking",
    "text",
    "usage",
    "latency_s",
    "time",
    "raw",
}
IDS = read_ids(RESULTS)
RUNS = sorted(p for p in RESULTS.glob("*") if p.is_dir() and any(p.glob("*/summary.json")))


def expected_folder(run_name: str) -> str:
    """results/<id>_<run>/; a run with no ID yet (a local mock run from before IDs) is still in results/<run>/."""
    return folder_name(IDS[run_name], run_name) if run_name in IDS else run_name


def model_summaries(run: Path) -> list[tuple[Path, dict]]:
    return [(p, json.loads(p.read_text())) for p in sorted(run.glob("*/summary.json"))]


def test_there_is_something_to_check():
    assert RUNS, "no run folders found under results/"


@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.name)
def test_run_has_a_valid_config_copy(run):
    copy = run / "config.yaml"
    assert copy.exists(), f"{copy} missing: results must come from scripts/run.py"
    cfg = yaml.safe_load(copy.read_text())
    RunConfig.model_validate(cfg)
    want = expected_folder(cfg["run"])
    assert run.name == want, f"{copy} says run: {cfg['run']!r}, so the folder should be {want!r}, not {run.name!r}"


@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.name)
def test_summaries_record_how_they_were_made(run):
    for path, s in model_summaries(run):
        missing = SUMMARY_FIELDS - set(s)
        assert not missing, f"{path} lacks {sorted(missing)}: rescore with scripts/run.py --no-calls"
        assert expected_folder(s["run"]) == run.name and len(s["config_sha256"]) == 64
        if s["n"]:
            missing = SCORED_FIELDS - set(s)
            assert not missing, f"{path} lacks {sorted(missing)}"
            assert len(s["per_item"]) == s["n"], f"{path}: per_item has {len(s['per_item'])} items, n is {s['n']}"


@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.name)
def test_replies_behind_each_summary_are_saved(run):
    for path, s in model_summaries(run):
        responses = path.parent / "responses.jsonl"
        rows = [json.loads(line) for line in responses.read_text().splitlines()] if responses.exists() else []
        request_keys = s.get("request_keys")
        current = [r for r in rows if
                   r["key"] == request_keys.get(r["id"].replace("constraints_", "image_", 1))
                   ] if request_keys is not None else [r for r in rows if r["key"] == s["key"]]
        assert len({r["id"] for r in current}) >= s["n"], f"{responses}: fewer saved replies than n={s['n']}"
        for r in current:
            missing = REPLY_FIELDS - set(r)
            assert not missing, f"{responses} row {r['id']} lacks {sorted(missing)}"
