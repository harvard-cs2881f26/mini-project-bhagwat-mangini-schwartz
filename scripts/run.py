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
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(open(args.config))
    rows = run(cfg, args.limit, args.models, args.workers)

    print(f"\n{'model':28s} {'n':>4s} {'risk acc':>9s} {'avg err s':>10s} {'≤0.5s':>7s} {'miss':>6s} {'false al':>9s} {'parse fail':>10s} {'errors':>6s}")
    for r in rows:
        if not r.get("n"):
            print(f"{r['name']:28s}    0  (no replies; errors={r.get('errors', 0)})")
            continue
        print(f"{r['name']:28s} {r['n']:4d} {fmt(r['risk_accuracy'], 1):>9s} {fmt(r['timing_mae_s']):>10s} "
              f"{fmt(r['within_0_5_rate'], 1):>7s} {fmt(r['miss_rate'], 1):>6s} {fmt(r['false_alarm_rate'], 1):>9s} "
              f"{fmt(r['parse_failure_rate'], 1):>10s} {r['errors']:6d}")
    print(f"\nSaved under results/{cfg['run']}/")


if __name__ == "__main__":
    main()
