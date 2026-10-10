"""Run a config: build requests -> call models -> save every reply -> score.

results/<run>/<model>/responses.jsonl   one line per reply, append-only, committed
results/<run>/<model>/summary.json      metrics + per-video outcomes
results/<run>/summary.json              one row per model

A reply is reused if one already exists for the same (video, model, prompt, thinking), so
reruns are free. Failed calls are printed and never saved.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from asimov.bench.video import build_request, load_prompt
from asimov.bench.video_score import score
from asimov.data.video import load_video_answers, load_video_items
from asimov.models import get_client

ROOT = Path(__file__).resolve().parents[2]
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def video_order(items, seed: int = 0):
    """Fixed shuffled order, so --limit N gives a mix of Yes/No videos and grows consistently."""
    items = sorted(items, key=lambda i: i.id)
    random.Random(seed).shuffle(items)
    return items


def run_key(model: str, prompt_name: str, thinking: str) -> str:
    h = hashlib.sha256(f"{model}|{load_prompt(prompt_name)}|{thinking}".encode()).hexdigest()
    return h[:12]


def load_saved(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return {r["id"]: r for r in rows if r["key"] == key}


def run_model(spec: dict, cfg: dict, items, out_dir: Path, workers: int, show: bool) -> dict:
    provider, model, thinking, prompt = spec["provider"], spec["model"], cfg["thinking"], cfg["prompt"]
    key = run_key(model, prompt, thinking)
    mdir = out_dir / model
    mdir.mkdir(parents=True, exist_ok=True)
    resp_path = mdir / "responses.jsonl"
    saved = load_saved(resp_path, key)
    todo = [i for i in items if i.id not in saved]
    print(f"\n== {spec['name']} ({model}): {len(items)} videos, {len(items) - len(todo)} saved, {len(todo)} to call")

    errors = 0
    if todo:
        client = get_client(provider, model)

        def call(item):
            return item, client.generate(build_request(item, prompt), thinking=thinking)

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
        return {"name": spec["name"], "model": model, "n": 0, "errors": errors}
    s = score(load_video_answers(), replies)
    usage = [saved[i]["usage"] for i in replies]
    summary = {"name": spec["name"], "provider": provider, "model": model, "prompt": prompt,
               "thinking": thinking, "errors": errors,
               "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
               "output_tokens": sum(u.get("output_tokens") or 0 for u in usage),
               **{k: v for k, v in s.items() if k != "per_item"}}
    (mdir / "summary.json").write_text(json.dumps({**summary, "per_item": s["per_item"]}, indent=1))
    return summary


def run(cfg: dict, limit: int | None = None, only: list[str] | None = None,
        workers: int = 4, results_dir: Path | None = None) -> list[dict]:
    items = video_order(load_video_items())[:limit]
    out_dir = (results_dir or ROOT / "results") / cfg["run"]
    show = limit is not None and limit <= 5
    rows = []
    for spec in cfg["models"]:
        if only and spec["name"] not in only and spec["model"] not in only:
            continue
        var = KEY_VARS.get(spec["provider"])
        if var and not os.environ.get(var):
            print(f"\n== {spec['name']}: skipped, {var} not set")
            continue
        rows.append(run_model(spec, cfg, items, out_dir, workers, show))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1))
    return rows
