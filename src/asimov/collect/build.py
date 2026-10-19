"""Everything collect.py writes, as {path under results/: text}. Pure: same runs in, same bytes out.

Run pages go in each run's own folder (results/<id>_<run>/). Never its writeup.md: that is hand-written, and only
read (into the run page).
"""

from __future__ import annotations

import yaml

from asimov.claims import Claims
from asimov.collect import pages
from asimov.collect.load import RunData
from asimov.collect.run_ids import RunLabel, assign_ids, run_labels
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
# Keys of one comparison. Optional: note (shown on the page); models (A's model id -> B's, when the ids differ;
# default: the same id).
COMPARISON_KEYS = {"name", "a", "b", "note", "models"}


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
    run_ids: dict[str, str] | None = None,
) -> dict[str, str]:
    """`run_ids` is results/run_ids.yaml (run name -> id); runs missing from it get fresh IDs. Descriptions are generated."""
    headline = headline or default_headline(runs)
    labels = run_labels(runs, assign_ids(run_ids or {}, runs))
    rows = index_rows(runs, prices, labels)
    index = {(r["run"], r["model"]): r for r in rows}
    files = {
        "index.csv": to_csv(rows, LEADING),
        "per_item.csv": to_csv(per_item_rows(runs, labels), PER_ITEM_COLUMNS),
        "README.md": pages.summary_page(runs, index, headline, claims, labels),
    }
    for run in runs:
        files[f"{run.folder}/README.md"] = pages.run_page(run, index, claims, labels, runs, headline)
        if pages.has_paper_check(run, claims):
            files[f"{run.folder}/paper_check.md"] = pages.paper_check_page(run, claims)
            files[f"{run.folder}/table.csv"] = to_csv(pages.check_csv_rows(run, claims), ())
    files |= comparison_pages(runs, comparisons, labels)
    return files


def comparison_pages(runs: list[RunData], comparisons: list[dict], labels: dict[str, RunLabel]) -> dict[str, str]:
    """results/compare/<name>.md per comparison."""
    by_name = {r.name: r for r in runs}
    files = {}
    for comp in comparisons:
        unknown = set(comp) - COMPARISON_KEYS
        if unknown:
            raise ValueError(
                f"comparison {comp.get('name')!r}: unknown key(s) {sorted(unknown)}; allowed: {sorted(COMPARISON_KEYS)}"
            )
        a, b = by_name.get(comp["a"]), by_name.get(comp["b"])
        if a is None or b is None:
            missing = [x for x in (comp["a"], comp["b"]) if x not in by_name]
            raise ValueError(f"comparison {comp['name']!r}: no results for run(s) {missing}; fix results/comparisons.yaml")
        mcs = pages.model_comparisons(a, b, comp.get("models"))
        files[f"compare/{comp['name']}.md"] = pages.comparison_page(comp["name"], a, b, comp.get("note") or "", labels, mcs)
    return files
