#!/usr/bin/env python
"""Run an eval config.

    python scripts/run.py configs/repro_fig5.yaml --limit 1 --models gpt-5-nano
    python scripts/run.py configs/repro_fig5.yaml --limit 5
    python scripts/run.py configs/repro_fig5.yaml            # all 287
    python scripts/run.py configs/smoke_mock.yaml            # free, no keys
    python scripts/run.py configs/repro_fig5.yaml --no-calls # rescore saved replies, free

--limit N   first N items of a fixed shuffled order (with N <= 5, every reply is printed)
--models    names or model IDs from the config (default: all whose API key is set)
--no-calls  make no API calls (no keys needed): rescore the saved replies and rewrite summaries
--no-collect  don't rebuild the results pages afterwards (scripts/collect.py; skipped for mock-only runs)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import collect  # noqa: E402  (scripts/collect.py)
from asimov.bench import get_benchmark  # noqa: E402
from asimov.collect.load import SKIP_PROVIDERS  # noqa: E402
from asimov.runner import run  # noqa: E402


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-calls", action="store_true")
    ap.add_argument("--no-collect", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    rows = run(cfg, args.limit, args.models, args.workers, config_path=config_path, call_models=not args.no_calls)
    cols = get_benchmark(cfg["benchmark"]).columns

    print(f"\n{'model':28s} {'n':>4s} " + " ".join(f"{label:>{len(label) + 1}s}" for label, _, _ in cols) + f" {'errors':>6s}")
    for r in rows:
        if not r.get("n"):
            print(f"{r['name']:28s}    0  (no replies; errors={r.get('errors', 0)})")
            continue
        cells = " ".join(f"{fmt(r[key], pct):>{len(label) + 1}s}" for label, key, pct in cols)
        print(f"{r['name']:28s} {r['n']:4d} {cells} {r['errors']:6d}")
    print(f"\nSaved under results/{cfg['run']}/")
    if not args.no_collect and any(m["provider"] not in SKIP_PROVIDERS for m in cfg["models"]):
        collect.main([])  # results pages stay in step with the replies (docs/PLAN.md §11)


if __name__ == "__main__":
    main()
