"""Everything collect.py writes, as {path under results/: text}. Pure: same runs in, same bytes out."""

from __future__ import annotations

import yaml

from asimov.claims import Claims
from asimov.collect import pages
from asimov.collect.load import RunData
from asimov.collect.run_ids import RunLabel, update_labels
from asimov.collect.tables import LEADING, PER_ITEM_COLUMNS, index_rows, per_item_rows, to_csv

HEADLINE_HEADER = """\
# Hand-edited (docs/PLAN.md §11): the official run per figure, and owners' notes shown on results/README.md.
# scripts/collect.py writes this file once, when it's missing, and never overwrites it.
"""
COMPARISONS_DEFAULT = """\
# Hand-edited (docs/PLAN.md §11): pairs of runs to compare on the same items. scripts/collect.py renders
# results/compare/<name>.md for each. Example:
#   - {name: fig4_min_vs_default, a: repro_fig4, b: repro_fig4_default, note: thinking min vs default}
comparisons: []
"""


def default_headline(runs: list[RunData]) -> dict:
    """The latest run of each figure's benchmark, with empty notes."""
    figures = {}
    for fig, (benchmark, _, _) in pages.FIGURES.items():
        candidates = sorted((pages.scored_date(r), r.name) for r in runs if r.benchmark == benchmark)
        figures[fig] = {"run": candidates[-1][1] if candidates else None, "note": ""}
    return {"figures": figures, "runs": {}}


def headline_yaml(runs: list[RunData]) -> str:
    return HEADLINE_HEADER + yaml.safe_dump(default_headline(runs), sort_keys=False)


def build(
    runs: list[RunData],
    claims: Claims,
    prices: dict,
    headline: dict,
    comparisons: list[dict],
    labels: dict[str, RunLabel] | None = None,
) -> dict[str, str]:
    """`labels` (results/run_ids.yaml) gives each run its ID and description; runs missing from it get fresh IDs."""
    headline = headline or default_headline(runs)
    labels = update_labels(labels or {}, runs)
    rows = index_rows(runs, prices, labels)
    index = {(r["run"], r["model"]): r for r in rows}
    files = {
        "index.csv": to_csv(rows, LEADING),
        "per_item.csv": to_csv(per_item_rows(runs, labels), PER_ITEM_COLUMNS),
        "README.md": pages.summary_page(runs, index, headline, claims, labels),
    }
    for run in runs:
        has_claims = any(c.benchmark == run.benchmark for c in claims.claims)
        files[f"{run.name}/README.md"] = pages.run_page(run, index, has_claims, labels)
        if has_claims:
            files[f"{run.name}/paper_check.md"] = pages.paper_check_page(run, claims)
            files[f"{run.name}/table.csv"] = to_csv(pages.check_csv_rows(run, claims), ())
    by_name = {r.name: r for r in runs}
    for comp in comparisons:
        a, b = by_name.get(comp["a"]), by_name.get(comp["b"])
        if a is None or b is None:
            missing = [x for x in (comp["a"], comp["b"]) if x not in by_name]
            raise ValueError(f"comparison {comp['name']!r}: no results for run(s) {missing}; fix results/comparisons.yaml")
        files[f"compare/{comp['name']}.md"] = pages.comparison_page(comp["name"], a, b, comp.get("note") or "", labels)
    return files
