"""Router search (src/asimov/router_search.py, scripts/router_search.py) on synthetic data: no results/ needed."""

import csv
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from asimov.bench.text_score import QUESTIONS
from asimov.collect.load import ModelResult
from asimov.prices import Price
from asimov.router_search import (
    Candidate,
    Member,
    Metrics,
    Outcome,
    SearchConfig,
    agreement,
    alone,
    candidates,
    cross_validate,
    cross_validate_groups,
    d_cascade,
    disagree,
    fold_ids,
    member_from,
    metrics_on,
    parallel,
    parallel_per_question_mix,
    pareto,
    per_question_mix,
    perfect_router_bound,
    q1_matched,
    select,
    split_by_question,
    three_stage,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import router_search as script  # noqa: E402  # pyright: ignore[reportMissingImports]
from results_tree import text_outcome, write_model  # noqa: E402  # pyright: ignore[reportMissingImports]

ALL = [True] * 4
NONE = [False] * 4


def member(label, correct, cost, latency, pred=None, confidence=None, self_hosted=False) -> Member:
    n = len(correct)
    preds = np.array(pred if pred is not None else [["A"] * 4] * n, dtype=object)
    return Member(
        label,
        np.array(correct, dtype=bool),
        preds,
        np.full(n, cost, dtype=float),
        np.full(n, latency, dtype=float),
        None if confidence is None else np.array(confidence, dtype=float),
        self_hosted,
    )


@pytest.fixture
def dec() -> Member:
    """Decision model: right on everything but item 2; confidence lowest on items 1 and 2."""
    return member("Dec", [ALL, ALL, NONE, ALL], 0.001, 0.1, confidence=[0.9, 0.5, 0.2, 0.6])


@pytest.fixture
def llm() -> Member:
    return member("LLM", [ALL, NONE, ALL, ALL], 0.01, 2.0)


EVERY = np.arange(4)


# --- folds -----------------------------------------------------------------------------------------------------


def test_fold_assignment_is_deterministic_and_balanced():
    a, b = fold_ids(319, 5, seed=3), fold_ids(319, 5, seed=3)
    assert (a == b).all()
    assert not (a == fold_ids(319, 5, seed=4)).all()
    assert sorted(np.bincount(a)) == [63, 64, 64, 64, 64]


def test_every_item_is_held_out_exactly_once_per_repeat():
    n = 23
    seen: list[np.ndarray] = []

    def spy(train: np.ndarray) -> Outcome:
        seen.append(train)
        return alone(member("S", [ALL] * n, 0.0, 1.0))

    cv = cross_validate([Candidate("single", "S", False, spy)], n, folds=5, seeds=(0, 1), caps={"cost": (1.0,)})
    held = cv[("cost", 1.0)]
    assert len(held.picks) == 10
    for seed in (0, 1):
        tests = [np.setdiff1d(np.arange(n), t) for t in seen[seed * 5 : seed * 5 + 5]]
        assert sorted(np.concatenate(tests).tolist()) == list(range(n))


def test_a_configuration_is_chosen_on_training_items_only():
    """B is right only on fold 0's items, A on 3 other items. Chosen on training items, B wins every fold but fold 0;
    a selection that peeked at all 20 items would pick B everywhere. No fit ever sees a held-out item."""
    n = 20
    folds = fold_ids(n, 5, 7)
    others = np.flatnonzero(folds != 0)[:3]
    a = member("A", [ALL if k in others else NONE for k in range(n)], 0.0, 1.0)
    b = member("B", [ALL if folds[k] == 0 else NONE for k in range(n)], 0.0, 1.0)
    trained: list[tuple[str, np.ndarray]] = []

    def fit(m: Member):
        def run(train: np.ndarray) -> Outcome:
            trained.append((m.label, train))
            return alone(m)

        return run

    cands = [Candidate("single", "A", False, fit(a)), Candidate("single", "B", False, fit(b))]
    cv = cross_validate(cands, n, folds=5, seeds=(7,), caps={"cost": (1.0,)})
    complements = [np.flatnonzero(folds != k).tolist() for k in range(5)]
    assert trained and all(train.tolist() in complements for _, train in trained)
    assert {p.fold: p.candidate for p in cv[("cost", 1.0)].picks} == {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}


# --- families: hand-worked ------------------------------------------------------------------------------------


def test_single_system_is_its_own_replies(dec):
    o = alone(dec)
    m = metrics_on([o], EVERY)
    assert m.accuracy[0] == pytest.approx(0.75)
    assert m.cost[0] == pytest.approx(1.0)  # $0.001 per item = $1 per 1,000
    assert m.latency[0] == pytest.approx(0.1)


def test_d_cascade_sends_the_lowest_confidence_share_of_training_items(dec, llm):
    # median of the training confidences [0.2, 0.5, 0.6, 0.9] is 0.55: items 1 and 2 go to the LLM
    o = d_cascade(dec, llm, 0.5).fit(EVERY)
    assert o.correct.tolist() == [ALL, NONE, ALL, ALL]
    assert o.cost.tolist() == pytest.approx([0.001, 0.011, 0.011, 0.001])
    assert o.latency.tolist() == pytest.approx([0.1, 2.1, 2.1, 0.1])
    # trained on items 0 and 3 only: threshold = median(0.9, 0.6) = 0.75, so items 1, 2, 3 are sent on
    o = d_cascade(dec, llm, 0.5).fit(np.array([0, 3]))
    assert o.cost.tolist() == pytest.approx([0.001, 0.011, 0.011, 0.011])


def test_agreement_sends_exactly_the_disagreeing_items():
    pa = [["A", "B", "C", "D"], ["A", "B", "C", "D"], ["A", "B", "C", "D"], ["A", "", "C", "D"]]
    pb = [["A", "B", "C", "D"], ["B", "B", "C", "D"], ["A", "B", "C", "A"], ["A", "", "C", "D"]]
    a = member("A", [NONE] * 4, 0.001, 0.5, pred=pa)
    b = member("B", [ALL, ALL, NONE, ALL], 0.002, 1.0, pred=pb)
    assert disagree(a.pred, b.pred, "any").tolist() == [False, True, True, True]  # unanswered never agrees
    assert disagree(a.pred, b.pred, "q1").tolist() == [False, True, False, False]
    strong = member("S", [ALL] * 4, 0.01, 3.0)
    o = agreement(a, b, strong, "any").fit(EVERY)
    assert o.correct.tolist() == [ALL, ALL, ALL, ALL]  # kept from B on item 0, the strong LLM on 1-3
    assert o.cost.tolist() == pytest.approx([0.003, 0.013, 0.013, 0.013])  # both first calls + the escalation
    assert o.latency.tolist() == pytest.approx([1.0, 4.0, 4.0, 4.0])  # first calls in parallel, then the strong one
    o = agreement(a, b, strong, "q1").fit(EVERY)
    assert o.correct.tolist() == [ALL, ALL, NONE, ALL]
    assert o.cost.tolist() == pytest.approx([0.003, 0.013, 0.003, 0.003])


def test_three_stage_escalates_only_low_confidence_items_where_the_cheap_llm_disagrees(dec):
    cheap = member("Cheap", [NONE, NONE, NONE, ALL], 0.002, 1.0, pred=[["A"] * 4, ["A"] * 4, ["B", "A", "A", "A"], ["A"] * 4])
    strong = member("Strong", [ALL] * 4, 0.01, 3.0)
    o = three_stage(dec, cheap, strong, 0.5, "any").fit(EVERY)
    # items 1, 2 go to the cheap LLM; it disagrees with Dec only on item 2, which goes to the strong LLM
    assert o.correct.tolist() == [ALL, NONE, ALL, ALL]
    assert o.cost.tolist() == pytest.approx([0.001, 0.003, 0.013, 0.001])
    assert o.latency.tolist() == pytest.approx([0.1, 1.1, 4.1, 0.1])


def test_per_question_mix_keeps_the_decision_answer_where_training_shows_it_better(dec):
    llm = member("LLM", [[False, True, True, True], [False, True, True, True], [False, True, True, True], ALL], 0.01, 2.0)
    o = per_question_mix(dec, llm, 0.5).fit(EVERY)
    # sent items 1, 2: Dec 1/2 right on Q1 vs LLM 0/2, so Q1 stays with Dec; Q2-Q4 go to the LLM
    assert o.correct.tolist() == [ALL, ALL, [False, True, True, True], ALL]
    assert o.cost.tolist() == pytest.approx([0.001, 0.011, 0.011, 0.001])


# --- selection ------------------------------------------------------------------------------------------------


def test_budget_selection_respects_the_cap_and_breaks_ties_by_cost():
    m = Metrics(np.array([0.9, 0.8, 0.8, 0.5]), np.array([5.0, 2.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0, 0.2]))
    assert select(m, "cost", 10.0) == 0
    assert select(m, "cost", 4.0) == 2  # 0.8 at $1 beats 0.8 at $2
    assert select(m, "cost", 0.5) == 3
    assert select(m, "latency", 0.5) == 3
    assert select(Metrics(np.array([0.9]), np.array([5.0]), np.array([1.0])), "cost", 1.0) is None


def test_pareto_keeps_only_undominated_points():
    acc, x = [0.5, 0.6, 0.55, 0.7, 0.7], [0.0, 1.0, 2.0, 3.0, 4.0]
    assert pareto(acc, x) == [0, 1, 3]


# --- script end to end ----------------------------------------------------------------------------------------

ITEMS = [f"t{k}" for k in range(10)]


def write_run(results: Path, run: str, model: str, correct: list[list[bool]], probs: dict | None):
    run_dir = results / run
    per_item = {i: text_outcome(c) for i, c in zip(ITEMS, correct, strict=True)}
    write_model(run_dir, model, model, "openai", "text", per_item, {"errors": 0}, run=run)
    path = run_dir / model / "responses.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    extra = {"usage": {"input_tokens": 1000, "output_tokens": 100}} | ({"probs": probs} if probs else {})
    path.write_text("".join(json.dumps({**r, **extra}) + "\n" for r in rows))
    cfg = {
        "run": run,
        "benchmark": "text",
        "prompt": "p",
        "thinking": "min",
        "models": [{"name": model, "provider": "openai", "model": model}],
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    results = tmp_path / "results"
    rng = np.random.default_rng(0)
    probs = {q: {"A": 0.7, "B": 0.3} for q in QUESTIONS}
    write_run(results, "dec_run", "dec-1", rng.random((10, 4)).round().astype(bool).tolist(), probs)
    write_run(results, "llm_run", "cheap-1", rng.random((10, 4)).round().astype(bool).tolist(), None)
    write_run(results, "llm_run2", "strong-1", [ALL] * 10, None)
    config = tmp_path / "router_search.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "decision": [{"run": "dec_run", "model": "dec-1", "label": "Dec"}],
                "cumulative_runs": "*_cumulative",
                "llms": [
                    {"run": "llm_run", "model": "cheap-1", "label": "Cheap"},
                    {"run": "llm_run2", "model": "strong-1", "label": "Strong"},
                ],
                "cheap": ["Cheap"],
                "strong": ["Strong"],
                "confidence": "min_top_probability",
                "fractions": [0.5],
                "agreement_rules": ["any", "q1"],
                "budgets_per_1000": [0.1, 100.0],
                "latency_caps_s": [1.0, 100.0],
                "folds": 5,
                "seeds": [0, 1],
            }
        )
    )
    prices = tmp_path / "prices.yaml"
    token = {"input_per_mtok": 1.0, "output_per_mtok": 1.0}
    prices.write_text(
        yaml.safe_dump(
            {
                "checked": "2026-10-12",
                "sources": {},
                "models": {
                    "dec-1": {"self_hosted": "GPU"},
                    "cheap-1": token,
                    "strong-1": {"input_per_mtok": 10.0, "output_per_mtok": 10.0},
                },
            }
        )
    )
    return {"results": results, "figures": tmp_path / "figures", "config": config, "prices": prices}


def args(tree: dict[str, Path]) -> list[str]:
    return [f"--{k}={v}" for k, v in tree.items()]


def test_search_config_rejects_unknown_keys_and_unknown_tiers(tree):
    cfg = yaml.safe_load(tree["config"].read_text())
    with pytest.raises(ValueError):
        SearchConfig.model_validate(cfg | {"threshold": 0.5})
    with pytest.raises(ValueError, match="not in llms"):
        SearchConfig.model_validate(cfg | {"strong": ["Nope"]})


def test_script_writes_tables_and_figures_then_check_passes(tree):
    assert script.main(args(tree)) == 0
    out = tree["results"] / "router_search"
    assert {p.name for p in out.iterdir()} == {"candidates.csv", "picks.csv", "pick_frequency.csv", "frontier.csv", "tables.md"}
    tables = (out / "tables.md").read_text()
    assert tables.splitlines()[2].startswith("**EXPLORATORY")
    assert "in-sample, optimistic, not a result" in tables
    for name in ("accuracy_vs_cost", "accuracy_vs_latency"):
        assert (tree["figures"] / "router_search" / f"{name}.png").exists()
    assert script.main([*args(tree), "--check"]) == 0
    (out / "picks.csv").write_text("edited\n")
    assert script.main([*args(tree), "--check"]) == 1


def test_script_adds_cumulative_runs_without_errors_and_skips_the_rest(tree):
    probs = {q: {"A": 0.6, "B": 0.4} for q in QUESTIONS}
    write_run(tree["results"], "part2_dec_cumulative", "dec-1", [ALL] * 10, probs)
    assert script.main(args(tree)) == 0
    names = (tree["results"] / "router_search" / "candidates.csv").read_text()
    assert "dec-1 (part2_dec_cumulative)" in names
    path = tree["results"] / "part2_dec_cumulative" / "dec-1" / "summary.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"errors": 1}))
    assert script.main(args(tree)) == 0
    tables = (tree["results"] / "router_search" / "tables.md").read_text()
    assert "skipped" in tables and "part2_dec_cumulative" in tables


# --- Modal compute time and pool notes ------------------------------------------------------------------------


def test_member_latency_is_modal_compute_time_where_a_row_has_it():
    ids = ["a", "b"]
    model = ModelResult(
        summary={"name": "m", "model": "m", "errors": 0},
        per_item={i: text_outcome(ALL) for i in ids},
        latencies=(5.0, 6.0),
        served_models=(),
        replies={},
        usage={i: {"input_tokens": 10, "output_tokens": 0} for i in ids},
        item_latency={"a": 5.0, "b": 6.0},
        item_compute_s={"a": 0.2},
    )
    m = member_from("M", model, Price(self_hosted="Modal L40S"), ids, None)
    assert m.latency.tolist() == [0.2, 6.0]  # compute_s where the row has it, else the round trip
    assert m.compute_rows == 1
    plain = member_from("P", replace(model, item_compute_s={}), Price(self_hosted="Modal L40S"), ids, None)
    assert plain.latency.tolist() == [5.0, 6.0] and plain.compute_rows == 0


def test_script_uses_compute_time_and_prints_pool_notes(tree):
    path = tree["results"] / "llm_run" / "cheap-1" / "responses.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(r | {"raw": {"remote": {"compute_s": 0.01}}}) + "\n" for r in rows))
    cfg = yaml.safe_load(tree["config"].read_text())
    cfg["llms"][0]["note"] = "11% parse failures"
    tree["config"].write_text(yaml.safe_dump(cfg))
    assert script.main(args(tree)) == 0
    out = tree["results"] / "router_search"
    rows = list(csv.DictReader(io.StringIO((out / "candidates.csv").read_text())))
    single = next(r for r in rows if r["candidate"] == "Cheap")
    assert float(single["latency_p50_s"]) == 0.01
    tables = (out / "tables.md").read_text()
    assert "Cheap: 11% parse failures" in tables
    assert "Cheap (10 of 10 items)" in tables  # which members' latency is Modal compute time


# --- extensions: split by question, parallel calls, the bound, Q1-matched, GPU time ----------------------------


def test_split_by_question_takes_each_question_from_its_side_and_pays_both_calls(dec, llm):
    o = split_by_question(dec, llm, (True, False, False, False)).fit(EVERY)
    # Q1 from Dec ([T, T, F, T] on the 4 items), Q2-Q4 from the LLM ([T, F, T, T])
    assert o.correct.tolist() == [ALL, [True, False, False, False], [False, True, True, True], ALL]
    assert o.cost.tolist() == pytest.approx([0.011] * 4)
    assert o.latency.tolist() == pytest.approx([2.1] * 4)  # sequential: the sum
    only_dec = split_by_question(dec, llm, (True,) * 4).fit(EVERY)
    assert only_dec.cost.tolist() == pytest.approx([0.001] * 4)  # one side answers nothing: its call isn't made
    assert only_dec.latency.tolist() == pytest.approx([0.1] * 4)
    only_llm = split_by_question(dec, llm, (False,) * 4).fit(EVERY)
    assert only_llm.cost.tolist() == pytest.approx([0.01] * 4)


def test_parallel_pays_both_calls_on_every_item_and_waits_for_the_slower(dec, llm):
    o = parallel(dec, llm, 0.5).fit(EVERY)
    # items 1 and 2 (lowest confidence) take the LLM's answers; the others keep Dec's
    assert o.correct.tolist() == [ALL, NONE, ALL, ALL]
    assert o.cost.tolist() == pytest.approx([0.011] * 4)
    assert o.latency.tolist() == pytest.approx([2.0] * 4)  # max(0.1, 2.0)


def test_parallel_per_question_mix_keeps_better_questions_and_pays_both(dec):
    llm = member("LLM", [[False, True, True, True], [False, True, True, True], [False, True, True, True], ALL], 0.01, 2.0)
    o = parallel_per_question_mix(dec, llm, 0.5).fit(EVERY)
    assert o.correct.tolist() == [ALL, ALL, [False, True, True, True], ALL]
    assert o.cost.tolist() == pytest.approx([0.011] * 4)
    assert o.latency.tolist() == pytest.approx([2.0] * 4)


def test_candidate_members_name_every_system_used(dec, llm):
    assert split_by_question(dec, llm, (True, False, False, False)).members == ("Dec", "LLM")
    assert parallel(dec, llm, 0.5).members == ("Dec", "LLM")
    strong = member("S", [ALL] * 4, 0.01, 3.0)
    assert three_stage(dec, llm, strong, 0.5, "any").members == ("Dec", "LLM", "S")


def test_candidates_add_14_splits_and_both_parallel_families_per_pair(dec, llm):
    cands = candidates([dec], [llm], [llm], [llm], (0.5,), ("any",))
    by_family: dict[str, int] = {}
    for c in cands:
        by_family[c.family] = by_family.get(c.family, 0) + 1
    assert by_family["split_by_question"] == 14  # the 2 one-sided splits are the single systems
    assert by_family["parallel"] == 1 and by_family["parallel_per_question_mix"] == 1


def test_gpu_seconds_count_self_hosted_calls_only_and_add_up_in_parallel():
    hosted = member("H", [ALL] * 4, 0.0, 0.3, confidence=[0.9, 0.5, 0.2, 0.6], self_hosted=True)
    api = member("A", [ALL] * 4, 0.01, 2.0)
    assert alone(hosted).gpu.tolist() == pytest.approx([0.3] * 4)
    assert alone(api).gpu.tolist() == [0.0] * 4
    assert parallel(hosted, api, 0.5).fit(EVERY).gpu.tolist() == pytest.approx([0.3] * 4)
    both = member("H2", [ALL] * 4, 0.0, 0.5, self_hosted=True)
    assert parallel(hosted, both, 0.5).fit(EVERY).gpu.tolist() == pytest.approx([0.8] * 4)  # GPU time is paid twice


def test_perfect_router_bound_sends_only_items_where_the_llm_gains(dec, llm):
    sent, acc, cost = perfect_router_bound(dec, llm)
    # only item 2 gains (Dec 0/4, LLM 4/4); item 1 would lose
    assert sent.tolist() == [0, 1]
    assert acc.tolist() == pytest.approx([0.75, 1.0])
    assert cost.tolist() == pytest.approx([1.0, 3.5])  # ($0.004 + $0.01) / 4 items x 1,000


def test_q1_matched_scores_q2_to_q4_on_items_whose_q1_is_right():
    one = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1]], dtype=float)
    n, acc = q1_matched(one[None])
    assert n == 2
    assert acc == pytest.approx({"latent_severity": 0.5, "effect": 0.0, "activated_risk": 1.0})
    # two repeats pool their matched items: 2 + 1 matched, mean 1.5 per repeat
    other = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=float)
    n, acc = q1_matched(np.stack([one, other]))
    assert n == 1.5
    assert acc["effect"] == pytest.approx(1 / 3)
    n, acc = q1_matched(np.zeros((1, 2, 4)))
    assert n == 0 and acc == dict.fromkeys(QUESTIONS[1:])


def test_group_selection_picks_within_each_group_only():
    n = 10
    good = member("Good", [ALL] * n, 0.0, 1.0)
    bad = member("Bad", [NONE] * n, 0.0, 1.0)
    cands = [Candidate("single", "Good", False, lambda t: alone(good)), Candidate("other", "Bad", False, lambda t: alone(bad))]
    groups = {"all": np.array([0, 1]), "other": np.array([1])}
    cv = cross_validate_groups(cands, n, 5, (0,), {"cost": (1.0,)}, groups)
    assert {p.candidate for p in cv[("all", "cost", 1.0)].picks} == {0}
    assert {p.candidate for p in cv[("other", "cost", 1.0)].picks} == {1}
    assert cross_validate(cands, n, 5, (0,), {"cost": (1.0,)})[("cost", 1.0)].picks == cv[("all", "cost", 1.0)].picks


def test_script_flags_noted_systems_everywhere_and_marks_every_table_exploratory(tree):
    cfg = yaml.safe_load(tree["config"].read_text())
    cfg["llms"][1]["note"] = "held for review"
    tree["config"].write_text(yaml.safe_dump(cfg))
    assert script.main(args(tree)) == 0
    out = tree["results"] / "router_search"
    lines = (out / "tables.md").read_text().splitlines()
    # every table (a header row followed by a --- row) has a bold EXPLORATORY line right above it
    headers = [k for k in range(1, len(lines)) if lines[k].startswith("| ---")]
    assert headers and all(lines[k - 3].startswith("**EXPLORATORY") for k in headers)
    rows_naming = [ln for ln in lines if ln.startswith("| ") and "Strong" in ln and not ln.startswith("| ---")]
    assert rows_naming and all("held for review" in ln for ln in rows_naming)
    frontier = list(csv.DictReader(io.StringIO((out / "frontier.csv").read_text())))
    assert any(r["set"] == "bound" for r in frontier)
    assert "bound, not a system" in "\n".join(lines)
    picks = list(csv.DictReader(io.StringIO((out / "picks.csv").read_text())))
    assert {"held_out_n_q1_matched", "held_out_effect_q1_matched", "held_out_gpu_s_per_item", "pool"} <= set(picks[0])
    assert {"split_by_question", "parallel"} <= {r["pool"] for r in picks}
    assert "EXPLORATORY" in script.figure_title("cost") and "EXPLORATORY" in script.figure_title("latency")


def test_script_reports_gpu_seconds_and_parameters_for_self_hosted_rows_never_dollars(tree):
    prices = yaml.safe_load(tree["prices"].read_text())
    prices["models"]["dec-1"] = {"self_hosted": "GPU", "params_b": 4}
    tree["prices"].write_text(yaml.safe_dump(prices))
    assert script.main(args(tree)) == 0
    lines = (tree["results"] / "router_search" / "tables.md").read_text().splitlines()
    dec_row = next(ln for ln in lines if ln.startswith("| Dec |"))
    assert "$" not in dec_row and "| 4 |" in dec_row and "round trip" in dec_row
