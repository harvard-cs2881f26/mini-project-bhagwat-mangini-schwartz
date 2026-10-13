#!/usr/bin/env python
"""Copy saved replies from one run into another, for models whose request is identical.

    python scripts/reuse_replies.py configs/repro_fig4.yaml configs/repro_fig4_default.yaml
    python scripts/reuse_replies.py <from config> <to config> --models claude-opus-4-5-20251101

Only models whose client ignores every setting that differs between the two configs are copied
(asimov/reuse.py); naming another model with --models is an error. Copied rows are marked
`reused_from`. Then score the target run without calls:
    python scripts/run.py <to config> --no-calls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.reuse import reuse  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_config")
    ap.add_argument("dst_config")
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    src = yaml.safe_load(Path(args.src_config).read_text())
    dst = yaml.safe_load(Path(args.dst_config).read_text())
    copied = reuse(src, dst, only=args.models)
    for model, n in copied.items():
        print(f"{model}: {n} replies copied from results/{src['run']}/ to results/{dst['run']}/")
    if not copied:
        print("nothing reusable: every model's request differs between the two configs")
    print(f"\nNext: python scripts/run.py {args.dst_config} --no-calls")


if __name__ == "__main__":
    main()
