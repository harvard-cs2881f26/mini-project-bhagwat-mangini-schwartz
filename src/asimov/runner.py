"""Run a config: build requests -> call models -> save every reply -> score.

The config's `benchmark:` picks the loader, request builder and scorer (asimov.bench); this
file is the same for every benchmark.

results/<run>/<model>/responses.jsonl   one line per reply, append-only, committed
results/<run>/<model>/summary.json      metrics + per-item outcomes + how they were made
results/<run>/summary.json              one row per model
results/<run>/config.yaml               exact copy of the config; one run name = one config

A reply is reused if one already exists for the same (item, model, prompt, thinking), so
reruns are free. Failed calls are printed and never saved. With call_models=False nothing is
called: saved replies are rescored (free, no keys needed).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from asimov.bench import Benchmark, get_benchmark
from asimov.config import RunConfig
from asimov.models import get_client

ROOT = Path(__file__).resolve().parents[2]
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


class ConfigChanged(RuntimeError):
    """results/<run>/ already holds results made with a different config."""


def fixed_order(items, seed: int = 0):
    """Fixed shuffled order, so --limit N gives a mix of answers and grows consistently."""
    items = sorted(items, key=lambda i: i.id)
    random.Random(seed).shuffle(items)
    return items


def run_key(model: str, prompt_version: str, thinking: str) -> str:
    """prompt_version: from Benchmark.prompt_version, e.g. the prompt file's full text."""
    h = hashlib.sha256(f"{model}|{prompt_version}|{thinking}".encode()).hexdigest()
    return h[:12]


def guard_config(out_dir: Path, config_text: str) -> None:
    """Write results/<run>/config.yaml, or stop if it exists with different contents."""
    copy = out_dir / "config.yaml"
    if copy.exists() and copy.read_text() != config_text:
        raise ConfigChanged(
            f"{copy} was written by a different config. Results made with different settings must "
            "not share a run name: give this config a new `run:` name (e.g. repro_fig4_default)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    copy.write_text(config_text)


def git_state() -> tuple[str | None, bool | None]:
    """(HEAD commit, uncommitted changes outside results/); (None, None) outside a git checkout."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", ".", ":!results"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def provenance(cfg: dict, config_text: str, config_path: Path | None) -> dict:
    """How a run's summaries were made; copied into every summary.json (with the model's key)."""
    commit, dirty = git_state()
    path = config_path.resolve() if config_path else None
    return {"benchmark": cfg["benchmark"], "run": cfg["run"],
            "config_path": str(path.relative_to(ROOT)) if path and path.is_relative_to(ROOT) else (str(path) if path else None),
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "git_commit": commit, "git_dirty": dirty, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def load_saved(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return {r["id"]: r for r in rows if r["key"] == key}


def run_model(bench: Benchmark, spec: dict, cfg: dict, items, out_dir: Path, workers: int, show: bool,
              made: dict, call_models: bool = True) -> dict:
    provider, model, thinking, prompt = spec["provider"], spec["model"], cfg["thinking"], cfg["prompt"]
    key = run_key(model, bench.prompt_version(prompt), thinking)
    mdir = out_dir / model
    mdir.mkdir(parents=True, exist_ok=True)
    resp_path = mdir / "responses.jsonl"
    saved = load_saved(resp_path, key)
    todo = [i for i in items if i.id not in saved]
    to_call = f"{len(todo)} to call" if call_models else "no calls"
    print(f"\n== {spec['name']} ({model}): {len(items)} items, {len(items) - len(todo)} saved, {to_call}")

    errors = 0
    if todo and call_models:
        client = get_client(provider, model)

        def call(item):
            return item, client.generate(bench.build_request(item, prompt), thinking=thinking)

        with ThreadPoolExecutor(max_workers=workers) as pool, open(resp_path, "a") as f:
            futures = [pool.submit(call, i) for i in todo]
            for fut in as_completed(futures):
                try:
                    item, r = fut.result()
                except Exception as e:  # never saved; rerun to retry
                    errors += 1
                    print(f"  ERROR {type(e).__name__}: {str(e)[:200]}")
                    continue
                row = {"id": item.id, "key": key, "provider": provider, "model": model,
                       "served_model": r.model, "prompt": prompt, "thinking": thinking,
                       "text": r.text, "usage": r.usage, "latency_s": round(r.latency_s, 2),
                       "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": r.raw}
                f.write(json.dumps(row) + "\n")
                f.flush()
                saved[item.id] = row
                if show:
                    print(f"  {item.id}  {r.latency_s:.1f}s  {r.usage}\n    reply: {r.text.strip()[:500]}")

    replies = {i.id: saved[i.id]["text"] for i in items if i.id in saved}
    if not replies:
        return {"name": spec["name"], "model": model, **made, "key": key, "n": 0, "errors": errors}
    s = bench.score(bench.load_answers(), replies)
    usage = [saved[i]["usage"] for i in replies]
    summary = {"name": spec["name"], "provider": provider, "model": model, "prompt": prompt,
               "thinking": thinking, **made, "key": key, "errors": errors,
               "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
               "output_tokens": sum(u.get("output_tokens") or 0 for u in usage),
               **{k: v for k, v in s.items() if k != "per_item"}}
    (mdir / "summary.json").write_text(json.dumps({**summary, "per_item": s["per_item"]}, indent=1))
    return summary


def run(cfg: dict, limit: int | None = None, only: list[str] | None = None,
        workers: int = 4, results_dir: Path | None = None,
        config_path: Path | None = None, call_models: bool = True) -> list[dict]:
    """config_path: the YAML file cfg was read from; its exact text is kept as results/<run>/config.yaml."""
    RunConfig.model_validate(cfg)  # unknown or missing keys fail here, before anything runs
    bench = get_benchmark(cfg["benchmark"])
    items = fixed_order(bench.load_items())[:limit]
    out_dir = (results_dir or ROOT / "results") / cfg["run"]
    config_text = config_path.read_text() if config_path else yaml.safe_dump(cfg, sort_keys=False)
    guard_config(out_dir, config_text)
    made = provenance(cfg, config_text, config_path)
    show = limit is not None and limit <= 5
    for name in only or []:
        if not any(name in (s["name"], s["model"]) for s in cfg["models"]):
            print(f"WARNING: {name!r} is not in this config. Options: {', '.join(s['model'] for s in cfg['models'])}")
    rows = []
    for spec in cfg["models"]:
        if only and spec["name"] not in only and spec["model"] not in only:
            continue
        var = KEY_VARS.get(spec["provider"])
        if call_models and var and not os.environ.get(var):
            print(f"\n== {spec['name']}: skipped, {var} not set")
            continue
        rows.append(run_model(bench, spec, cfg, items, out_dir, workers, show, made, call_models))
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1))
    return rows
