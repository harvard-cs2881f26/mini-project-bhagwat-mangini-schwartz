#!/usr/bin/env python
"""Compare two saved text runs of one model: a prompt variant against its baseline, on the items both have.

    python scripts/qwen_format_check.py            # writes results/<config stem>/tables.md
    python scripts/qwen_format_check.py --check    # exit 1 if tables.md differs from a fresh rebuild

EXPLORATORY ($0, saved replies only). Config: configs/analysis/qwen_format_check.yaml. Reads each
run's summary.json (`per_item`, as scored) and responses.jsonl (rows under the summary's key) for
reply text, finish, output tokens and `raw.remote.compute_s`. Numbers only:
- parse failures and their causes (a failing item can have several): an "E" on a 4-option question
  (Q2-Q4), another invalid value, cut off at the token cap, a missing key in a complete reply, no JSON;
- strict accuracy per question (unanswered = wrong) and Q2-Q4 on the items whose Q1 matched the key,
  with Wilson 95% CIs;
- the paired bootstrap difference in each item's mean of the 4 questions (variant - baseline), and
  exact McNemar per question (src/asimov/stats.py);
- output tokens and compute_s, p50 / p95.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.bench.parse import json_object, loose_pairs  # noqa: E402
from asimov.bench.text_score import QUESTIONS, locate  # noqa: E402
from asimov.run_ids import folder_name, read_ids  # noqa: E402
from asimov.stats import mcnemar_exact, paired_bootstrap_diff, wilson_ci  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "analysis" / "qwen_format_check.yaml"
CAUSES = ("invalid E", "invalid other", "cut off", "missing key", "no JSON")
SEVERITY = QUESTIONS[1:]  # Q2-Q4 have options A-D only


class RunRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run: str
    model: str


class CheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    baseline: RunRef
    variant: RunRef


def load_run(results: Path, ref: RunRef) -> tuple[str, dict, dict]:
    """(run folder, per_item, rows by id under the summary's key)."""
    folder = folder_name(read_ids(results)[ref.run], ref.run)
    mdir = results / folder / ref.model
    summary = json.loads((mdir / "summary.json").read_text())
    rows = [json.loads(line) for line in (mdir / "responses.jsonl").read_text().splitlines() if line.strip()]
    return folder, summary["per_item"], {r["id"]: r for r in rows if r["key"] == summary["key"]}


def raw_value(text: str, question: str) -> object:
    """The value the reply wrote for a question's answer, as the scorer finds it (None if absent)."""
    answer = json_object(text)
    if answer is None or not any(kind == "answer" for _, kind in locate(answer)):
        answer = loose_pairs(text)
    found = locate(answer or {}).get((question, "answer"))
    return (answer or {})[found[0]] if found else None


def causes(item: dict, row: dict) -> set[str]:
    """Why a failing item's answers couldn't all be read (several can apply)."""
    out = set()
    for q in QUESTIONS:
        status = item["parse"][q]
        if status == "invalid_letter":
            value = raw_value(row["text"], q)
            out.add("invalid E" if q in SEVERITY and isinstance(value, str) and value.strip().upper() == "E" else "invalid other")
        elif status == "missing_key" and item["finish"] != "truncated":
            out.add("missing key")
        elif status == "no_json" and item["finish"] != "truncated":
            out.add("no JSON")
    if item["finish"] == "truncated":
        out.add("cut off")
    return out


def pct(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def quantiles(values: list[float], fmt: str) -> str:
    p50, p95 = np.percentile(values, [50, 95])
    return f"{p50:{fmt}} / {p95:{fmt}}"


def tables(cfg: CheckConfig, results: Path, config_path: str) -> str:
    runs = {name: load_run(results, ref) for name, ref in (("baseline", cfg.baseline), ("variant", cfg.variant))}
    shared = sorted(set(runs["baseline"][1]) & set(runs["variant"][1]) & set(runs["baseline"][2]) & set(runs["variant"][2]))
    n = len(shared)
    names = {k: f"{v[0]} ({k})" for k, v in runs.items()}
    out = [
        f"# Qwen format check: {names['variant']} vs {names['baseline']}",
        "",
        f"**{cfg.label}**",
        "",
        f"Built by `scripts/qwen_format_check.py --config {config_path}` from saved replies ($0). {n} items in both runs.",
        "",
    ]

    out += [
        "## Parse failures and causes",
        "",
        "A failing item can have several causes.",
        "",
        "| Run | Parse fail | " + " | ".join(CAUSES) + " |",
        "|---|---|" + "---|" * len(CAUSES),
    ]
    for k, (_, per_item, rows) in runs.items():
        failing = [i for i in shared if not per_item[i]["parse_ok"]]
        counts = {c: sum(c in causes(per_item[i], rows[i]) for i in failing) for c in CAUSES}
        out.append(
            f"| {names[k]} | {len(failing)} ({len(failing) / n:.1%}) | " + " | ".join(str(counts[c]) for c in CAUSES) + " |"
        )

    out += [
        "",
        "## Accuracy (Wilson 95% CI)",
        "",
        "Strict: unanswered = wrong, all items. Q1-matched: Q2-Q4 on the items whose "
        "Q1 matched the key, within each run (the subsets differ between runs).",
        "",
        "| Run | Q1 | Q2 | Q3 | Q4 | Q1 answered E (key E) | Q1 matched (n) | Q2 matched | Q3 matched | Q4 matched |",
        "|---|" + "---|" * 9,
    ]
    for k, (_, per_item, _) in runs.items():
        strict = [pct(sum(per_item[i][q]["correct"] for i in shared), n) for q in QUESTIONS]
        matched = [i for i in shared if per_item[i][QUESTIONS[0]]["correct"]]
        m = [pct(sum(per_item[i][q]["correct"] for i in matched), len(matched)) for q in SEVERITY]
        q1 = [per_item[i][QUESTIONS[0]] for i in shared]
        said_e = f"{sum(o['pred'] == 'E' for o in q1)} ({sum(o['gold'] == 'E' for o in q1)})"
        out.append(f"| {names[k]} | " + " | ".join(strict) + f" | {said_e} | {len(matched)} | " + " | ".join(m) + " |")

    base, var = runs["baseline"][1], runs["variant"][1]
    mean4 = {k: [np.mean([v[i][q]["correct"] for q in QUESTIONS]) for i in shared] for k, v in (("b", base), ("v", var))}
    diff, lo, hi = paired_bootstrap_diff(mean4["v"], mean4["b"])
    out += [
        "",
        "## Paired comparison (variant - baseline)",
        "",
        f"Mean of the 4 questions per item: baseline {np.mean(mean4['b']):.3f}, variant {np.mean(mean4['v']):.3f}, "
        f"difference {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] (paired bootstrap, 10,000 resamples, seed 0).",
        "",
        "| Question | variant right, baseline wrong | baseline right, variant wrong | McNemar p (exact) |",
        "|---|---|---|---|",
    ]
    for q in QUESTIONS:
        b = sum(var[i][q]["correct"] and not base[i][q]["correct"] for i in shared)
        c = sum(base[i][q]["correct"] and not var[i][q]["correct"] for i in shared)
        out.append(f"| {q} | {b} | {c} | {mcnemar_exact(b, c):.3g} |")

    out += [
        "",
        "## Cost per item",
        "",
        "| Run | Output tokens p50 / p95 | Input tokens p50 | compute_s p50 / p95 | Cut off (all replies) |",
        "|---|---|---|---|---|",
    ]
    for k, (_, _, rows) in runs.items():
        r = [rows[i] for i in shared]
        out.append(
            f"| {names[k]} | {quantiles([x['usage']['output_tokens'] for x in r], '.0f')} | "
            f"{np.percentile([x['usage']['input_tokens'] for x in r], 50):.0f} | "
            f"{quantiles([x['raw']['remote']['compute_s'] for x in r], '.1f')} | "
            f"{sum(runs[k][1][x['id']]['finish'] == 'truncated' for x in r)} |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--check", action="store_true", help="compare tables.md with a fresh rebuild instead of writing")
    args = ap.parse_args(argv)
    cfg = CheckConfig.model_validate(yaml.safe_load(args.config.read_text()))
    shown = args.config.resolve()
    text = tables(cfg, args.results, str(shown.relative_to(ROOT)) if shown.is_relative_to(ROOT) else args.config.name)
    path = args.results / args.config.stem / "tables.md"
    if args.check:
        if not path.exists() or path.read_text() != text:
            print(f"stale: {path}")
            return 1
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
