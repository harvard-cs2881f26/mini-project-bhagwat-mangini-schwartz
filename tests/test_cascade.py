"""asimov.cascade (Part 2 designs B and D, simulated from saved replies) and scripts/cascade.py on a tiny results tree."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from asimov.bench.text_score import QUESTIONS
from asimov.cascade import (
    METRICS,
    DesignConfig,
    ItemResult,
    System,
    accuracy,
    at_fraction,
    cascade_curve,
    check_pair,
    item_confidence,
    perfect_router,
    split_table,
    system_from,
)
from asimov.collect.load import ModelResult
from asimov.prices import Price
from asimov.stats import bootstrap_mean_ci, paired_bootstrap_diff, wilson_ci

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cascade as cascade_script  # noqa: E402  # pyright: ignore[reportMissingImports]
from results_tree import text_outcome, write_model  # noqa: E402  # pyright: ignore[reportMissingImports]

ALL = (True, True, True, True)
NONE = (False, False, False, False)


def item(correct, cost=0.001, latency=1.0) -> ItemResult:
    return ItemResult(tuple(correct), cost, latency)


@pytest.fixture
def pair() -> tuple[System, System]:
    """A decision model right on its confident items, an LLM right on everything but i4."""
    first = System(
        "Dec",
        {
            "i1": item(ALL, 0.0, 0.1),
            "i2": item((True, False, False, False), 0.0, 0.1),
            "i3": item(NONE, 0.0, 0.1),
            "i4": item(ALL, 0.0, 0.1),
        },
        {"i1": 0.9, "i2": 0.5, "i3": 0.2, "i4": 0.5},
    )
    llm = System(
        "LLM",
        {"i1": item(ALL, 0.01, 2.0), "i2": item(ALL, 0.01, 2.0), "i3": item(ALL, 0.01, 2.0), "i4": item(NONE, 0.01, 2.0)},
        {},
    )
    return first, llm


def test_confidence_is_the_lowest_top_probability_across_the_questions():
    probs = {
        "latent_risk": {"A": 0.9, "B": 0.1},
        "latent_severity": {"A": 0.3, "B": 0.7},
        "effect": {"A": 0.6, "B": 0.4},
        "activated_risk": {"C": 0.8, "D": 0.2},
    }
    assert item_confidence(probs) == pytest.approx(0.6)


def test_a_question_without_probabilities_gives_confidence_zero():
    probs = {q: {"A": 1.0} for q in QUESTIONS[1:]}
    assert item_confidence(probs) == 0.0
    assert item_confidence(probs | {"latent_risk": {}}) == 0.0
    assert item_confidence(None) == 0.0


def test_curve_endpoints_are_the_two_systems_alone(pair):
    first, llm = pair
    curve = cascade_curve(first, llm)
    start, end = curve[0], curve[-1]
    assert (start.n_sent, end.n_sent) == (0, 4)
    assert start.accuracy["latent_risk"].value == 0.75 and start.accuracy["effect"].value == 0.5
    assert end.accuracy["latent_risk"].value == 0.75 and end.accuracy["effect"].value == 0.75
    assert start.cost_per_1000 == pytest.approx(0.0)
    assert end.cost_per_1000 == pytest.approx(10.0)  # $0.01 per item
    assert start.latency_p50 == pytest.approx(0.1) and end.latency_p50 == pytest.approx(2.1)


def test_items_are_sent_on_in_order_of_confidence_and_ties_go_together(pair):
    first, llm = pair
    curve = cascade_curve(first, llm)
    assert [p.n_sent for p in curve] == [0, 1, 3, 4]  # 0.2, then both 0.5s, then 0.9
    assert [p.threshold for p in curve] == [0.2, 0.5, 0.9, float("inf")]
    assert curve[1].accuracy["mean"].value == pytest.approx((4 + 1 + 4 + 4) / 16)


def test_fixed_fractions_send_exactly_that_share_lowest_confidence_first_ties_by_id(pair):
    first, llm = pair
    points = {f: at_fraction(first, llm, f) for f in (0.25, 0.5, 0.75)}
    assert [p.n_sent for p in points.values()] == [1, 2, 3]
    assert points[0.5].accuracy["mean"].value == 1.0  # i3 then i2 (tied with i4 at 0.5; i2 first by id)
    assert points[0.75].accuracy["mean"].value == pytest.approx(12 / 16)
    assert at_fraction(first, llm, 0.0).n_sent == 0 and at_fraction(first, llm, 1.0).n_sent == 4


def test_perfect_router_sends_only_items_with_a_gain_and_beats_the_curve(pair):
    first, llm = pair
    for metric in METRICS:
        frontier = perfect_router(first, llm, metric)
        assert frontier[0].n_sent == 0
        best = frontier[-1].accuracy[metric].value
        for p in cascade_curve(first, llm):
            assert best >= p.accuracy[metric].value
            same = [f for f in frontier if f.n_sent == min(p.n_sent, frontier[-1].n_sent)]
            assert same[0].accuracy[metric].value >= p.accuracy[metric].value
    mean = perfect_router(first, llm, "mean")
    assert [p.n_sent for p in mean] == [0, 1, 2]  # i3 (gain 4), then i2 (gain 3); never i1 or i4
    assert mean[-1].accuracy["mean"].value == 1.0
    assert set(mean[-1].accuracy) == {"mean"}


def test_split_table_has_16_rows_with_the_single_systems_as_endpoints(pair):
    first, llm = pair
    rows = split_table(first, llm, headline=("latent_risk",))
    assert len(rows) == 16
    by_split = {r.decision_questions: r for r in rows}
    alone, none = by_split[QUESTIONS], by_split[()]
    assert alone.accuracy["effect"].value == 0.5 and none.accuracy["effect"].value == 0.75
    assert alone.cost_per_1000 == pytest.approx(0.0) and none.cost_per_1000 == pytest.approx(10.0)
    assert [r.decision_questions for r in rows if r.headline] == [("latent_risk",)]
    headline = by_split[("latent_risk",)]
    assert headline.accuracy["latent_risk"].value == 0.75 and headline.accuracy["latent_severity"].value == 0.75
    assert headline.cost_per_1000 == pytest.approx(10.0)  # one LLM call still answers all 4 questions
    assert none.latency_p50 == pytest.approx(2.0) and headline.latency_p50 == pytest.approx(2.1)  # the LLM alone makes one call
    ids = sorted(first.items)
    mean = lambda s: [sum(s.items[i].correct) / 4 for i in ids]  # noqa: E731
    headline_mean = [(first.items[i].correct[0] + sum(llm.items[i].correct[1:])) / 4 for i in ids]
    assert headline.diff_vs_llm is not None and none.diff_vs_llm is None
    assert (headline.diff_vs_llm.value, headline.diff_vs_llm.lo, headline.diff_vs_llm.hi) == pytest.approx(
        paired_bootstrap_diff(headline_mean, mean(llm))
    )


def test_cis_come_from_stats():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]
    a = accuracy(scores, "effect")
    assert (a.lo, a.hi) == pytest.approx(wilson_ci(3, 5))
    m = accuracy([1.0, 0.25, 0.5], "mean")
    assert (m.value, m.lo, m.hi) == pytest.approx(bootstrap_mean_ci([1.0, 0.25, 0.5]))


def test_a_decision_item_missing_from_the_llm_run_stops_the_analysis(pair):
    first, llm = pair
    short = System("LLM", {k: v for k, v in llm.items.items() if k != "i3"}, {})
    with pytest.raises(ValueError, match="i3"):
        check_pair(first, short)


def test_a_scored_item_without_saved_usage_or_latency_stops_the_analysis():
    per_item = {"i1": {q: {"correct": True} for q in QUESTIONS}}
    summary = {"name": "M", "model": "m", "errors": 0}
    no_usage = ModelResult(summary, per_item, (), (), {}, {}, {}, {"i1": 1.0})
    no_latency = ModelResult(summary, per_item, (), (), {}, {"i1": {}}, {}, {})
    for model, field in ((no_usage, "usage"), (no_latency, "latency")):
        with pytest.raises(ValueError, match=f"no saved {field}"):
            system_from(model, Price(free_tier=True))


# --- scripts/cascade.py end to end ---------------------------------------------------------------------------

ITEMS = ["t1", "t2", "t3", "t4"]
PROBS = {q: {"A": 0.7, "B": 0.3} for q in QUESTIONS}


def write_run(results: Path, run: str, model: str, provider: str, correct: list[list[bool]], extra: dict, errors: int = 0):
    run_dir = results / run
    per_item = {i: text_outcome(c) for i, c in zip(ITEMS, correct, strict=True)}
    write_model(run_dir, model, model, provider, "text", per_item, {"errors": errors}, run=run)
    path = run_dir / model / "responses.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps({**r, **extra}) + "\n" for r in rows))
    cfg = {
        "run": run,
        "benchmark": "text",
        "prompt": "p",
        "thinking": "min",
        "models": [{"name": model, "provider": provider, "model": model}],
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    results = tmp_path / "results"
    write_run(results, "llm_run", "gpt-5", "openai", [[True] * 4] * 4, {"usage": {"input_tokens": 1000, "output_tokens": 100}})
    write_run(
        results, "dec_run", "dec-1", "openai", [[True, False, True, False]] * 4, {"usage": {"input_tokens": 500}, "probs": PROBS}
    )
    config = tmp_path / "designs.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "escalation": [{"run": "llm_run", "model": "gpt-5"}],
                "first": [{"run": "dec_run", "model": "dec-1"}],
                "confidence": "min_top_probability",
                "fixed_fractions": [0.25, 0.5, 0.75],
                "split_headline": ["latent_risk"],
            }
        )
    )
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        yaml.safe_dump(
            {
                "checked": "2026-10-12",
                "sources": {},
                "models": {"gpt-5": {"input_per_mtok": 1.25, "output_per_mtok": 10.0}, "dec-1": {"free_tier": True}},
            }
        )
    )
    return {"results": results, "figures": tmp_path / "figures", "config": config, "prices": prices}


def args(tree: dict[str, Path]) -> list[str]:
    return [f"--{k}={v}" for k, v in tree.items()]


def test_design_config_rejects_unknown_keys(tree):
    cfg = yaml.safe_load(tree["config"].read_text()) | {"threshold": 0.5}
    with pytest.raises(ValueError):
        DesignConfig.model_validate(cfg)


def test_script_writes_tables_and_figures_then_check_passes(tree):
    assert cascade_script.main(args(tree)) == 0
    out = tree["results"] / "designs"
    assert {p.name for p in out.iterdir()} == {
        "cascade.csv",
        "perfect_router.csv",
        "fixed_fractions.csv",
        "splits.csv",
        "tables.md",
    }
    assert (tree["figures"] / "designs" / "cascade_dec-1_to_gpt-5.png").exists()
    tables = (out / "tables.md").read_text()
    assert "n = 4" in tables and "smoke run, not a result" not in tables
    assert cascade_script.main([*args(tree), "--check"]) == 0
    (out / "splits.csv").write_text("edited\n")
    assert cascade_script.main([*args(tree), "--check"]) == 1


def test_script_stops_when_a_run_has_failed_calls(tree):
    summary_path = tree["results"] / "dec_run" / "dec-1" / "summary.json"
    summary = json.loads(summary_path.read_text()) | {"errors": 2}
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="failed calls"):
        cascade_script.main(args(tree))


def test_script_stamps_runs_smaller_than_the_llm_run(tree):
    path = tree["results"] / "dec_run" / "dec-1" / "summary.json"
    summary = json.loads(path.read_text())
    summary["per_item"].pop("t4")
    summary["n"] = 3
    path.write_text(json.dumps(summary))
    assert cascade_script.main(args(tree)) == 0
    assert "smoke run, not a result" in (tree["results"] / "designs" / "tables.md").read_text()
