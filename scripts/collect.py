#!/usr/bin/env python
"""Rebuild the results tables and pages from results/ (docs/PLAN.md §11). Free: no API calls, no data/.

    python scripts/collect.py           # write results/index.csv, per_item.csv, README.md, <id>_<run>/README.md, ...
    python scripts/collect.py --check   # exit 1 if the committed pages differ from a fresh rebuild (CI)

Reads results/<id>_<run>/*/summary.json and responses.jsonl, configs/paper/claims.yaml, and
configs/prices.yaml if it exists. Hand-edited inputs: results/headline.yaml (created once if
missing), results/comparisons.yaml, and each run's results/<id>_<run>/writeup.md, which is included verbatim
in its run page and never written here: after editing a write-up, rerun this (--check fails until you do).
results/run_ids.yaml (run name -> short run ID) is maintained here:
new runs are added, existing IDs never change. Every other file it writes is generated: never edit it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.claims import CLAIMS_PATH, load_claims  # noqa: E402
from asimov.collect.build import COMPARISONS_DEFAULT, build, headline_yaml  # noqa: E402
from asimov.collect.load import load_results, load_yaml  # noqa: E402
from asimov.collect.pages import GENERATED  # noqa: E402
from asimov.collect.run_ids import assign_ids, missing_ids  # noqa: E402
from asimov.prices import PRICES_PATH, load_prices  # noqa: E402
from asimov.run_ids import RUN_IDS_FILE, parse_ids, run_ids_yaml  # noqa: E402

# Paths collect.py may have written earlier. A match is orphaned when the rebuild no longer makes it and it's
# ours: a page starting with the GENERATED marker, or a run's table.csv (hand-written files are left alone).
GENERATED_PATTERNS = ("compare/*.md", "*/README.md", "*/paper_check.md", "*/table.csv")


def stale_files(results_dir: Path, files: dict[str, str]) -> list[str]:
    return [p for p, text in sorted(files.items()) if not (results_dir / p).exists() or (results_dir / p).read_text() != text]


def orphaned_files(results_dir: Path, files: dict[str, str]) -> list[str]:
    """Generated files under results/ that the rebuild no longer produces (e.g. for a deleted run)."""
    found = {p for pattern in GENERATED_PATTERNS for p in results_dir.glob(pattern)}
    orphans = [
        p
        for p in found
        if p.relative_to(results_dir).as_posix() not in files and (p.name == "table.csv" or p.read_text().startswith(GENERATED))
    ]
    return sorted(p.relative_to(results_dir).as_posix() for p in orphans)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with a fresh rebuild instead of writing")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--claims", type=Path, default=CLAIMS_PATH)
    ap.add_argument("--prices", type=Path, default=PRICES_PATH)
    args = ap.parse_args(argv)

    runs = load_results(args.results)
    headline_path, comparisons_path = args.results / "headline.yaml", args.results / "comparisons.yaml"
    if not args.check:
        if not headline_path.exists():
            headline_path.write_text(headline_yaml(runs))
        if not comparisons_path.exists():
            comparisons_path.write_text(COMPARISONS_DEFAULT)
    headline = load_yaml(headline_path, {})
    comparisons = load_yaml(comparisons_path, {}).get("comparisons") or []
    run_ids_path = args.results / RUN_IDS_FILE
    existing = parse_ids(load_yaml(run_ids_path, {}))
    ids = assign_ids(existing, runs)
    prices = load_prices(args.prices).models if args.prices.exists() else {}
    files = build(runs, load_claims(args.claims), prices, headline, comparisons, ids)

    orphans = orphaned_files(args.results, files)
    if args.check:
        stale = stale_files(args.results, files)
        unlabeled = missing_ids(existing, runs)
        if unlabeled:
            stale.append(RUN_IDS_FILE)
            print(f"{RUN_IDS_FILE} is stale: runs without an ID: {', '.join(unlabeled)}")
        if stale:
            print("results pages are stale: run scripts/collect.py and commit. Differs: " + ", ".join(stale))
        if orphans:
            print("generated files no longer produced by a rebuild: delete them. Orphaned: " + ", ".join(orphans))
        if stale or orphans:
            return 1
        print(f"results pages are up to date ({len(files)} files)")
        return 0
    if ids != existing or not run_ids_path.exists():
        run_ids_path.write_text(run_ids_yaml(ids))
    for rel, text in files.items():
        path = args.results / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(f"collect: {len(runs)} runs, {len(files)} files under {args.results}")
    if orphans:
        print("collect: generated files no longer produced (delete them, --check fails on them): " + ", ".join(orphans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
