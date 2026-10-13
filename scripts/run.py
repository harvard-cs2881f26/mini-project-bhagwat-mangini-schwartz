#!/usr/bin/env python
"""Run an eval config.

    python scripts/run.py configs/repro_fig5.yaml --limit 1 --models gpt-5-nano
    python scripts/run.py configs/repro_fig5.yaml --limit 5
    python scripts/run.py configs/repro_fig5.yaml            # all 287
    python scripts/run.py configs/smoke_mock.yaml            # free, no keys

--limit N   first N videos of a fixed shuffled order (with N <= 5, every reply is printed)
--models    names or model IDs from the config (default: all whose API key is set)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.bench import get_benchmark  # noqa: E402
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
    ap.add_argument("--score-only", action="store_true", help="Rescore caches without API calls")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(open(args.config))
    rows = run(cfg, args.limit, args.models, args.workers, score_only=args.score_only)
    name = cfg.get("benchmark", "video")
    name = "image" if name == "constraints" else name
    cols = get_benchmark(cfg.get("variant", name) if name == "image" else name).columns

    print(f"\n{'model':28s} {'n':>4s} " + " ".join(f"{label:>{len(label) + 1}s}" for label, _, _ in cols) + f" {'errors':>6s}")
    for r in rows:
        if not r.get("n"):
            print(f"{r['name']:28s}    0  (no replies; errors={r.get('errors', 0)})")
            continue
        cells = " ".join(f"{fmt(r[key], pct):>{len(label) + 1}s}" for label, key, pct in cols)
        print(f"{r['name']:28s} {r['n']:4d} {cells} {r['errors']:6d}")
    print(f"\nSaved under results/{cfg['run']}/")


if __name__ == "__main__":
    main()
