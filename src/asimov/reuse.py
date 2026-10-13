"""Reuse saved replies across runs when a model's request is identical (CLI: scripts/reuse_replies.py).

Two configs can differ in settings that a given client never sends. Claude's client, for example,
sends the same request for `thinking: min` and `default`, because no thinking budget is sent. Its
replies from one run are then the replies the other run would get, so they're copied rather than
paid for again. Each copied row keeps its text and usage, gets the target run's cache key, and
records `reused_from` (source run and key). Summaries count them (`n_reused`).

A model is reusable only if the two configs share benchmark and prompt, its two specs have the same
service and request params (`runner.request_extra`: base_url, params; decision services share model
ids across services and variants), and every setting that differs between them is one its client
ignores (`models.request_settings`).
"""

from __future__ import annotations

import json
from pathlib import Path

from asimov.bench import get_benchmark
from asimov.config import RunConfig
from asimov.models import request_settings
from asimov.runner import ROOT, load_saved, request_extra, run_key

SETTINGS = ("thinking", "log_reasoning")  # config settings a client may or may not send


def differing_settings(src: dict, dst: dict) -> set[str]:
    return {s for s in SETTINGS if src.get(s, False) != dst.get(s, False)}


def reusable(provider: str, src: dict, dst: dict) -> bool:
    same_request = src["benchmark"] == dst["benchmark"] and src["prompt"] == dst["prompt"]
    return same_request and not (differing_settings(src, dst) & request_settings(provider))


def copy_rows(rows: list[dict], src_key: str, dst_key: str, src_run: str, already: set[str]) -> list[dict]:
    """Rows saved under `src_key`, re-keyed for the target run; ids already saved there are skipped."""
    return [
        {**r, "key": dst_key, "reused_from": {"run": src_run, "key": src_key}}
        for r in rows
        if r["key"] == src_key and r["id"] not in already
    ]


def reuse(src: dict, dst: dict, only: list[str] | None = None, results_dir: Path | None = None) -> dict[str, int]:
    """Copy reusable models' replies from src's run into dst's run. Returns model -> rows copied.
    Raises ValueError for a requested model whose request differs between the two configs."""
    RunConfig.model_validate(src)
    RunConfig.model_validate(dst)
    results = results_dir or ROOT / "results"
    bench = get_benchmark(dst["benchmark"])
    src_models = {m["model"]: m for m in src["models"]}
    copied = {}
    for spec in dst["models"]:
        model, provider = spec["model"], spec["provider"]
        if (only and model not in only and spec["name"] not in only) or model not in src_models:
            continue
        src_extra, dst_extra = request_extra(src_models[model]), request_extra(spec)
        if not reusable(provider, src, dst) or src_extra != dst_extra:
            if only:
                why = sorted(differing_settings(src, dst) & request_settings(provider))
                if src_extra != dst_extra:
                    why.append(f"service/params {src_extra!r} vs {dst_extra!r}")
                raise ValueError(f"{model}: request differs between {src['run']} and {dst['run']} ({why}); call it instead")
            continue
        version = bench.prompt_version(dst["prompt"])
        src_key = run_key(model, version, src["thinking"], src.get("log_reasoning", False), extra=src_extra)
        dst_key = run_key(model, version, dst["thinking"], dst.get("log_reasoning", False), extra=dst_extra)
        src_path = results / src["run"] / model / "responses.jsonl"
        dst_path = results / dst["run"] / model / "responses.jsonl"
        rows = [json.loads(line) for line in src_path.read_text().splitlines()] if src_path.exists() else []
        new = copy_rows(rows, src_key, dst_key, src["run"], already=set(load_saved(dst_path, dst_key)))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in new)
        copied[model] = len(new)
    return copied
