#!/usr/bin/env python
"""How often two saved text runs give the same answer, per question, on the items both have.

    python scripts/agreement.py results/part2_qwen_logprob/qwen3.5-4b results/part2_qwen_gen/qwen3.5-4b

Reads each model folder's summary.json (`per_item` predictions, as scored), never the answer key.
Numbers only. An unanswered question (no prediction) on either side counts as not agreeing, and
is also counted on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.bench.text_score import QUESTIONS  # noqa: E402


def agreement(a: dict[str, dict], b: dict[str, dict]) -> dict[str, dict[str, int]]:
    """Per question: items in both, same answer, unanswered in a, unanswered in b."""
    shared = sorted(set(a) & set(b))
    out = {}
    for q in QUESTIONS:
        pa, pb = [a[i][q]["pred"] for i in shared], [b[i][q]["pred"] for i in shared]
        out[q] = {
            "n": len(shared),
            "agree": sum(x is not None and x == y for x, y in zip(pa, pb, strict=True)),
            "unanswered_a": sum(x is None for x in pa),
            "unanswered_b": sum(y is None for y in pb),
        }
    return out


def per_item(model_dir: Path) -> dict[str, dict]:
    return json.loads((model_dir / "summary.json").read_text())["per_item"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path, help="results/<run>/<model>")
    ap.add_argument("b", type=Path, help="results/<run>/<model>")
    args = ap.parse_args()
    print(f"a = {args.a}\nb = {args.b}")
    for q, c in agreement(per_item(args.a), per_item(args.b)).items():
        print(f"{q:16} agree {c['agree']}/{c['n']}  unanswered a {c['unanswered_a']}, b {c['unanswered_b']}")


if __name__ == "__main__":
    main()
