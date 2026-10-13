"""Run a config: build requests -> call models -> save every reply -> score.

The config's `benchmark:` picks the loader, request builder and scorer (asimov.bench); this
file is the same for every benchmark.

results/<id>_<run>/<model>/responses.jsonl   one line per reply, append-only, committed
results/<id>_<run>/<model>/summary.json      metrics + per-item outcomes + how they were made
results/<id>_<run>/summary.json              one row per model
results/<id>_<run>/config.yaml               exact copy of the config; one run name = one config

A reply is reused if one already exists for the same (item, model, prompt, thinking, and
log_reasoning when on), so reruns are free. Failed calls are printed and logged to
results/<id>_<run>/<model>/errors.jsonl, never saved as replies, so a rerun retries them. With
call_models=False nothing is called: saved replies are rescored (free, no keys needed).
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
from asimov.models.reply_info import cached_input_tokens, finish
from asimov.run_ids import run_dir

ROOT = Path(__file__).resolve().parents[2]
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
            "openai_decisions": "OPENAI_API_KEY"}  # systemone models name theirs in the config (key_env)


class ConfigChanged(RuntimeError):
    """results/<id>_<run>/ already holds results made with a different config."""


def fixed_order(items, seed: int = 0):
    """Fixed shuffled order, so --limit N gives a mix of answers and grows consistently."""
    items = sorted(items, key=lambda i: i.id)
    random.Random(seed).shuffle(items)
    return items


def run_key(model: str, prompt_version: str, thinking: str, log_reasoning: bool = False, extra: str = "") -> str:
    """prompt_version: from Benchmark.prompt_version, e.g. the prompt file's full text.

    log_reasoning and extra (the service and request params, `request_extra`) change the request,
    so they change the key; when off or empty the key is the same as before they existed, so
    earlier replies stay valid.
    """
    request = f"{model}|{prompt_version}|{thinking}" + ("|log_reasoning" if log_reasoning else "")
    request += f"|{extra}" if extra else ""
    return hashlib.sha256(request.encode()).hexdigest()[:12]


def request_extra(spec: dict) -> str:
    """The parts of a model spec that change what is sent beyond the model id: which service
    (base_url) and extra params. "" for every LLM spec, so their keys are unchanged."""
    parts = [spec.get("base_url") or ""] + ([json.dumps(spec["params"], sort_keys=True)] if spec.get("params") else [])
    return "|".join(parts) if any(parts) else ""


def guard_config(out_dir: Path, config_text: str) -> None:
    """Write results/<id>_<run>/config.yaml, or stop if it exists with different settings.

    Settings are compared as parsed YAML, so editing a comment keeps the run name (and refreshes
    the copy); any change to a value, including the model list, needs a new run name.
    """
    copy = out_dir / "config.yaml"
    if copy.exists() and yaml.safe_load(copy.read_text()) != yaml.safe_load(config_text):
        raise ConfigChanged(
            f"{copy} was written by a different config. Results made with different settings must "
            "not share a run name: give this config a new `run:` name (e.g. repro_fig4_default)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    copy.write_text(config_text)


def git_state() -> tuple[str | None, bool | None]:
    """(HEAD commit, edits to tracked files outside results/); (None, None) outside a git checkout."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def provenance(cfg: dict, config_text: str, config_path: Path | None) -> dict:
    """How a run's summaries were made; copied into every summary.json (with the model's key)."""
    commit, dirty = git_state()
    path = config_path.resolve() if config_path else None
    return {"benchmark": cfg["benchmark"], "run": cfg["run"],
            "config_path": str(path.relative_to(ROOT)) if path and path.is_relative_to(ROOT) else (str(path) if path else None),
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "git_commit": commit, "git_dirty": dirty, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def last_errors(mdir: Path) -> int:
    """Failed calls in the model's last run that made calls, from its summary.json (kept by --no-calls).

    The count belongs to that run's items; a --no-calls run with another --limit still reports it.
    """
    path = mdir / "summary.json"
    return json.loads(path.read_text()).get("errors", 0) if path.exists() else 0


def load_saved(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return {r["id"]: r for r in rows if r["key"] == key}


def run_model(bench: Benchmark, spec: dict, cfg: dict, items, out_dir: Path, workers: int, show: bool,
              made: dict, call_models: bool = True) -> dict:
    provider, model, thinking, prompt = spec["provider"], spec["model"], cfg["thinking"], cfg["prompt"]
    log_reasoning = cfg.get("log_reasoning", False)
    key = run_key(model, bench.prompt_version(prompt), thinking, log_reasoning, request_extra(spec))
    mdir = out_dir / model
    mdir.mkdir(parents=True, exist_ok=True)
    resp_path = mdir / "responses.jsonl"
    saved = load_saved(resp_path, key)
    todo = [i for i in items if i.id not in saved]
    to_call = f"{len(todo)} to call" if call_models else "no calls"
    print(f"\n== {spec['name']} ({model}): {len(items)} items, {len(items) - len(todo)} saved, {to_call}")

    errors = 0 if call_models else last_errors(mdir)
    if todo and call_models:
        service = {k: spec[k] for k in ("base_url", "key_env", "params") if spec.get(k)}  # decision services only
        client = get_client(provider, model, **service)

        def call(item):
            return client.generate(bench.build_request(item, prompt), thinking=thinking, log_reasoning=log_reasoning)

        with ThreadPoolExecutor(max_workers=workers) as pool, open(resp_path, "a") as f:
            futures = {pool.submit(call, i): i for i in todo}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:  # logged to errors.jsonl, never saved as a reply; rerun to retry
                    errors += 1
                    log_error(mdir / "errors.jsonl", item.id, key, e)
                    print(f"  ERROR {type(e).__name__}: {str(e)[:200]}")
                    continue
                usage = {**r.usage, "cached_input_tokens": cached_input_tokens(provider, r.raw)}
                row = {"id": item.id, "key": key, "provider": provider, "model": model,
                       "served_model": r.model, "prompt": prompt, "thinking": thinking,
                       "log_reasoning": log_reasoning, "text": r.text, "reasoning": r.reasoning,
                       "usage": usage, "latency_s": round(r.latency_s, 2),
                       "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": r.raw}
                if r.probs:  # decision models only, so LLM rows stay exactly as before
                    row["probs"] = r.probs
                f.write(json.dumps(row) + "\n")
                f.flush()
                saved[item.id] = row
                if show:
                    print(f"  {item.id}  {r.latency_s:.1f}s  {r.usage}\n    reply: {r.text.strip()[:500]}")

    replies = {i.id: saved[i.id]["text"] for i in items if i.id in saved}
    if not replies:
        empty = {"name": spec["name"], "model": model, **made, "key": key, "n": 0, "errors": errors}
        (mdir / "summary.json").write_text(json.dumps(empty, indent=1))  # keeps the error count for --no-calls
        return empty
    s = bench.score(bench.load_answers(), replies)
    rows = [saved[i] for i in replies]
    finishes = {r["id"]: finish(provider, r["raw"]) for r in rows}
    per_item = {i: {**outcome, "finish": finishes[i]} for i, outcome in s.pop("per_item").items()}
    service = {k: spec[k] for k in ("base_url", "params") if spec.get(k)}  # decision services only
    summary = {"name": spec["name"], "provider": provider, "model": model, **service, "prompt": prompt,
               "thinking": thinking, "log_reasoning": log_reasoning, **made, "key": key, "errors": errors,
               **token_totals(provider, rows), **finish_rates(finishes),
               "n_with_reasoning": sum(bool(r.get("reasoning", "").strip()) for r in rows),
               "n_reused": sum("reused_from" in r for r in rows), **s}  # copied by asimov.reuse
    (mdir / "summary.json").write_text(json.dumps({**summary, "per_item": per_item}, indent=1))
    return summary


def log_error(path: Path, item_id: str, key: str, e: Exception) -> None:
    """A failed call, for the record only: the cache never reads this file, so it's retried."""
    row = {"id": item_id, "key": key, "error_type": type(e).__name__, "message": str(e)[:2000],
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def token_totals(provider: str, rows: list[dict]) -> dict:
    """Summed usage. Cached input tokens are read from the saved raw response, so old replies count too."""
    cached = [cached_input_tokens(provider, r["raw"]) for r in rows]
    return {"input_tokens": sum(r["usage"].get("input_tokens") or 0 for r in rows),
            "output_tokens": sum(r["usage"].get("output_tokens") or 0 for r in rows),
            "cached_input_tokens": None if any(c is None for c in cached) else sum(c or 0 for c in cached)}


def finish_rates(finishes: dict[str, str]) -> dict:
    """Share of replies cut off by the output limit, and refused or safety-blocked (models.reply_info)."""
    n = len(finishes)
    return {"truncation_rate": sum(f == "truncated" for f in finishes.values()) / n,
            "refusal_rate": sum(f == "refused" for f in finishes.values()) / n}


def run(cfg: dict, limit: int | None = None, only: list[str] | None = None,
        workers: int = 4, results_dir: Path | None = None,
        config_path: Path | None = None, call_models: bool = True) -> list[dict]:
    """config_path: the YAML file cfg was read from; its exact text is kept as results/<id>_<run>/config.yaml."""
    RunConfig.model_validate(cfg)  # unknown or missing keys and unknown providers fail here, before anything runs
    bench = get_benchmark(cfg["benchmark"])
    bench.prompt_version(cfg["prompt"])  # an unknown prompt name fails before results/<id>_<run>/ is touched
    items = fixed_order(bench.load_items())[:limit]
    providers = [m["provider"] for m in cfg["models"]]
    out_dir = run_dir(results_dir or ROOT / "results", cfg["run"], cfg["benchmark"], providers)
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
        var = spec.get("key_env") or KEY_VARS.get(spec["provider"])
        if call_models and var and not os.environ.get(var):
            print(f"\n== {spec['name']}: skipped, {var} not set")
            continue
        rows.append(run_model(bench, spec, cfg, items, out_dir, workers, show, made, call_models))
    write_combined(cfg, out_dir)
    return rows


def write_combined(cfg: dict, out_dir: Path) -> None:
    """results/<id>_<run>/summary.json: one row per config model, from each model's own summary.json,
    so running a subset (--models) never drops the other models' rows."""
    combined = []
    for spec in cfg["models"]:
        path = out_dir / spec["model"] / "summary.json"
        if path.exists():
            s = json.loads(path.read_text())
            s.pop("per_item", None)
            combined.append(s)
    (out_dir / "summary.json").write_text(json.dumps(combined, indent=1))
