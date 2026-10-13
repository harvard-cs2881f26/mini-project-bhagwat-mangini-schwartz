"""Run a config: build requests -> call models -> save every reply -> score.

results/<run>/<model>/responses.jsonl   one line per reply, append-only, committed
results/<run>/<model>/summary.json      metrics + per-item outcomes
results/<run>/summary.json              one row per model

Image requests are fingerprinted by prompt, image bytes, provider, model and thinking.
Video cache keys remain compatible. API errors are audited separately, never reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from asimov.bench import Benchmark, get_benchmark
from asimov.bench import image as image_bench
from asimov.models import get_client

ROOT = Path(__file__).resolve().parents[2]
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def minute_retry_delay(error: Exception) -> float | None:
    """Honor Gemini's explicit per-minute retry hint; never retry daily/account limits."""
    payload = getattr(error, "details", None)
    if not isinstance(payload, dict):
        return None
    details = payload.get("error", {}).get("details", [])
    violations = [v for d in details for v in d.get("violations", [])]
    if not violations or not all("PerMinute" in v.get("quotaId", "") for v in violations):
        return None
    for detail in details:
        if "RetryInfo" in detail.get("@type", ""):
            try:
                delay = float(detail["retryDelay"].removesuffix("s")) + 1
            except (ValueError, TypeError, KeyError):
                return None
            return delay if 0 <= delay <= 60 else None
    return None


def fixed_order(items, seed: int = 0):
    """Fixed shuffled order, so --limit N grows consistently across benchmarks."""
    items = sorted(items, key=lambda i: i.id)
    random.Random(seed).shuffle(items)
    return items


def run_key(model: str, prompt_version: str, thinking: str) -> str:
    h = hashlib.sha256(f"{model}|{prompt_version}|{thinking}".encode()).hexdigest()
    return h[:12]


def load_saved(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return {r["id"]: r for r in rows if r["key"] == key}


def run_model(bench: Benchmark, spec: dict, cfg: dict, items, out_dir: Path, workers: int, show: bool,
              score_only: bool = False) -> dict:
    provider, model, thinking, prompt = spec["provider"], spec["model"], cfg["thinking"], cfg.get("prompt", cfg.get("variant", "image"))
    benchmark = {"constraints": "image"}.get(cfg.get("benchmark"), cfg.get("benchmark", "video"))
    message_roles = cfg.get("message_roles", "user")
    if message_roles not in ("user", "system_user"):
        raise ValueError(f"Unknown message_roles: {message_roles}")
    if message_roles == "system_user" and (benchmark != "image" or provider != "openai"):
        raise ValueError("system_user currently supports only the OpenAI image benchmark")
    if benchmark == "image":
        variant = {"constraints": "image", "constraints_rationale": "image_rationale"}.get(cfg.get("variant"), cfg.get("variant", "image"))
        keys = {item.id: image_bench.request_key(item, provider, model, thinking, variant, message_roles)
                for item in items}
    else:
        key = run_key(model, bench.prompt_version(prompt), thinking)
        keys = {item.id: key for item in items}
    mdir = out_dir / model
    mdir.mkdir(parents=True, exist_ok=True)
    resp_path = mdir / "responses.jsonl"
    if benchmark == "image":
        saved = {}
        if resp_path.exists():
            for line in resp_path.read_text().splitlines():
                row = json.loads(line)
                item_id = row["id"].replace("constraints_", "image_", 1)
                if row["key"] == keys.get(item_id):
                    saved[item_id] = row
    else:
        saved = load_saved(resp_path, key)
    todo = [i for i in items if i.id not in saved]
    action = "missing (score-only; no calls)" if score_only else "to call"
    print(f"\n== {spec['name']} ({model}): {len(items)} items, {len(items) - len(todo)} saved, {len(todo)} {action}")

    errors = 0
    if todo and not score_only:
        client = get_client(provider, model)
        interval = float(spec.get("request_interval_s", 0))
        if not 0 <= interval <= 60:
            raise ValueError("request_interval_s must be between 0 and 60")
        pace_lock = threading.Lock()
        last_start = 0.0

        def call(item):
            nonlocal last_start
            system_kwargs = {}
            if message_roles == "system_user":
                system, parts = image_bench.build_system_request(item)
                system_kwargs["system"] = system
            else:
                parts = bench.build_request(item, prompt)
            for attempt in range(4):
                with pace_lock:
                    delay = max(0, interval - (time.monotonic() - last_start))
                    if delay:
                        time.sleep(delay)
                    last_start = time.monotonic()
                try:
                    return item, client.generate(parts, thinking=thinking, **system_kwargs)
                except Exception as error:
                    delay = minute_retry_delay(error)
                    if delay is None or attempt == 3:
                        raise
                    print(f"  {item.id}: per-minute rate limit; retry in {delay:.0f}s")
                    time.sleep(delay)

        with ThreadPoolExecutor(max_workers=workers) as pool, open(resp_path, "a") as f:
            pending_items = iter(todo)
            futures = {}
            quota_exhausted = False

            def submit_next():
                item = next(pending_items, None)
                if item is not None:
                    futures[pool.submit(call, item)] = item

            for _ in range(workers):
                submit_next()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    requested = futures.pop(fut)
                    try:
                        item, r = fut.result()
                    except Exception as e:  # never saved; rerun to retry
                        errors += 1
                        message = str(e)
                        # Audit only; this file is never read as a response cache.
                        details = json.dumps(getattr(e, "details", None), default=str)
                        for variable in KEY_VARS.values():
                            secret = os.environ.get(variable)
                            if secret:
                                message = message.replace(secret, "[redacted]")
                                details = details.replace(secret, "[redacted]")
                        print(f"  ERROR {requested.id} {type(e).__name__}: {message[:200]}")
                        with (mdir / "attempt_errors.jsonl").open("a") as error_log:
                            error_log.write(json.dumps({"id": requested.id, "model": model,
                                                        "type": type(e).__name__, "message": message,
                                                        "details": json.loads(details)}) + "\n")
                        if any(marker in message for marker in ("RESOURCE_EXHAUSTED", "insufficient_quota")):
                            quota_exhausted = True
                            print("  Quota exhausted: stopping new calls for this model.")
                        continue
                    row = {"id": item.id, "key": keys[item.id], "provider": provider, "model": model,
                           "served_model": r.model, "prompt": prompt, "benchmark": benchmark, "thinking": thinking,
                           "message_roles": message_roles,
                           "text": r.text, "usage": r.usage, "latency_s": round(r.latency_s, 2),
                           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": r.raw}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    saved[item.id] = row
                    if show:
                        print(f"  {item.id}  {r.latency_s:.1f}s  {r.usage}\n    reply: {r.text.strip()[:500]}")
                    elif len(saved) % 20 == 0:
                        print(f"  Saved {len(saved)}/{len(items)} replies")
                if not quota_exhausted:
                    for _ in done:
                        submit_next()

    replies = {i.id: saved[i.id]["text"] for i in items if i.id in saved}
    s = bench.score(bench.load_answers(), replies) if replies or benchmark == "image" else {"n": 0, "per_item": {}}
    usage = [saved[i]["usage"] for i in replies]
    summary = {"name": spec["name"], "provider": provider, "model": model, "prompt": prompt,
               "thinking": thinking, "benchmark": benchmark, "message_roles": message_roles, "errors": errors,
               "n_requested": len(items), "n_missing": len(items) - len(replies),
               "n_dataset": cfg.get("dataset_size", len(items)),
               "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
               "output_tokens": sum(u.get("output_tokens") or 0 for u in usage),
               **{k: v for k, v in s.items() if k != "per_item"}}
    (mdir / "summary.json").write_text(json.dumps({**summary, "per_item": s["per_item"]}, indent=1))
    return summary


def run(cfg: dict, limit: int | None = None, only: list[str] | None = None,
        workers: int = 4, results_dir: Path | None = None, score_only: bool = False) -> list[dict]:
    benchmark = {"constraints": "image"}.get(cfg.get("benchmark"), cfg.get("benchmark", "video"))
    if benchmark == "image_rationale":
        cfg = {**cfg, "benchmark": "image", "variant": "image_rationale"}
        benchmark = "image"
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    variant = {"constraints": "image", "constraints_rationale": "image_rationale"}.get(cfg.get("variant"), cfg.get("variant", "image"))
    if benchmark == "image" and variant not in ("image", "image_rationale"):
        raise ValueError(f"unknown constraints variant: {variant}")
    bench = get_benchmark(variant if benchmark == "image" else benchmark)
    loaded = bench.load_items()
    items = fixed_order(loaded)[:limit]
    out_dir = (results_dir or ROOT / "results") / cfg["run"]
    show = limit is not None and limit <= 5
    for name in only or []:
        if not any(name in (s["name"], s["model"]) for s in cfg["models"]):
            print(f"WARNING: {name!r} is not in this config. Options: {', '.join(s['model'] for s in cfg['models'])}")
    rows = []
    for spec in cfg["models"]:
        if only and spec["name"] not in only and spec["model"] not in only:
            continue
        var = KEY_VARS.get(spec["provider"])
        if var and not os.environ.get(var) and not score_only:
            print(f"\n== {spec['name']}: skipped, {var} not set")
            continue
        rows.append(run_model(bench, spec, {**cfg, "dataset_size": len(loaded)}, items, out_dir, workers, show, score_only=score_only))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1))
    return rows
