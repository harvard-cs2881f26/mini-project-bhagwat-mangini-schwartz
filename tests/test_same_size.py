"""asimov.same_size (same-size comparison, from saved replies) and scripts/same_size.py on a tiny results tree."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from asimov.bench.text_score import QUESTIONS
from asimov.cascade import MEAN, ItemResult, System
from asimov.collect.load import load_model
from asimov.prices import Price
from asimov.same_size import (
    COMPUTE,
    ROUND_TRIP,
    ROUND_TRIP_CUMULATIVE,
    ContrastSpec,
    RowSpec,
    SameSizeConfig,
    contrast_family,
    item_latencies,
    row_result,
)
from asimov.stats import holm, mcnemar_exact, paired_bootstrap_diff, wilson_ci

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import same_size as same_size_script  # noqa: E402  # pyright: ignore[reportMissingImports]

from results_tree import ITEMS, text_outcome, write_model  # noqa: E402  # pyright: ignore[reportMissingImports]

SELF_HOSTED = Price(self_hosted="Modal L40S", params_b=4)


def spec(label: str = "A", **kw) -> RowSpec:
    return RowSpec(run=kw.pop("run", "r"), model=kw.pop("model", "m"), label=label, group=kw.pop("group", "g"), **kw)


def cfg_dict() -> dict:
    return {
        "label": "EXPLORATORY",
        "rows": [
            {"run": "dec_run", "model": "dec-1", "label": "Dec", "group": "decision"},
            {"run": "gen_run", "model": "gen-1", "label": "Gen", "group": "generate", "flag": "high parse failures"},
        ],
        "contrasts": [{"a": "Dec", "b": "Gen", "kind": "pre-registered", "changes": "how the answer is produced"}],
    }


def test_config_rejects_unknown_keys_duplicate_labels_and_unknown_contrast_rows():
    SameSizeConfig.model_validate(cfg_dict())
    with pytest.raises(ValueError):
        SameSizeConfig.model_validate(cfg_dict() | {"extra": 1})
    dup = cfg_dict()
    dup["rows"][1]["label"] = "Dec"
    with pytest.raises(ValueError, match="duplicate"):
        SameSizeConfig.model_validate(dup)
    bad = cfg_dict()
    bad["contrasts"][0]["b"] = "Nope"
    with pytest.raises(ValueError, match="Nope"):
        SameSizeConfig.model_validate(bad)
    with pytest.raises(ValueError):
        ContrastSpec(a="Dec", b="Gen", kind="post-hoc", changes="x")  # pyright: ignore[reportArgumentType]


def test_latency_prefers_modal_compute_when_every_item_has_it():
    rt, comp = {"a": 2.0, "b": 3.0}, {"a": 0.2, "b": 0.3}
    assert item_latencies(["a", "b"], rt, comp, cumulative=False) == (comp, COMPUTE)
    assert item_latencies(["a", "b"], rt, {}, cumulative=False) == (rt, ROUND_TRIP)
    assert item_latencies(["a", "b"], rt, {}, cumulative=True) == (rt, ROUND_TRIP_CUMULATIVE)


def test_latency_stops_on_partial_timings():
    with pytest.raises(ValueError, match="compute_s"):
        item_latencies(["a", "b"], {"a": 1.0, "b": 1.0}, {"a": 0.1}, cumulative=False)
    with pytest.raises(ValueError, match="latency"):
        item_latencies(["a", "b"], {"a": 1.0}, {}, cumulative=False)


def write_run(results: Path, run: str, model: str, correct: list[list[bool]], extra: dict | None = None, **metrics):
    """A text run whose summary matches its per-item outcomes; `extra` is merged into every saved reply."""
    run_dir = results / run
    per_item = {i: text_outcome(c) for i, c in zip(ITEMS, correct, strict=True)}
    acc = {f"{q}_accuracy": sum(per_item[i][q]["correct"] for i in ITEMS) / len(ITEMS) for q in QUESTIONS}
    base = {"parse_failure_rate": 0.25, "n_q1_matched": 2, "latent_severity_accuracy_q1_matched": 0.5}
    write_model(run_dir, model, model, "openai", "text", per_item, acc | base | metrics, run=run)
    path = run_dir / model / "responses.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    usage = {"usage": {"input_tokens": 1000, "output_tokens": 100}}
    path.write_text("".join(json.dumps({**r, **usage, **(extra or {})}) + "\n" for r in rows))
    cfg = {
        "run": run,
        "benchmark": "text",
        "prompt": "p",
        "thinking": "min",
        "models": [{"name": model, "provider": "openai", "model": model}],
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    return load_model(run_dir / model)


DEC = [[True, True, False, True], [True, False, True, True], [False, True, True, False], [True, True, True, True]]
GEN = [[False, True, False, True], [False, False, False, True], [False, True, True, True], [True, False, True, False]]


def test_load_model_reads_modal_compute_seconds(tmp_path):
    m = write_run(tmp_path / "results", "gen_run", "gen-1", GEN, {"raw": {"remote": {"compute_s": 0.5}}})
    assert m.item_compute_s == dict.fromkeys(ITEMS, 0.5)
    assert set(m.item_latency) == set(ITEMS)  # round trip still kept
    plain = write_run(tmp_path / "results", "dec_run", "dec-1", DEC)
    assert plain.item_compute_s == {}


def test_row_result_matches_summary_and_prices(tmp_path):
    m = write_run(tmp_path / "results", "dec_run", "dec-1", DEC)
    r = row_result(spec("Dec"), m, Price(input_per_mtok=1.0, output_per_mtok=10.0), cumulative=False)
    assert r.n == 4
    for q in QUESTIONS:
        assert r.accuracy[q].value == m.summary[f"{q}_accuracy"]
    k = sum(c[0] for c in DEC)
    assert (r.accuracy["latent_risk"].lo, r.accuracy["latent_risk"].hi) == wilson_ci(k, 4)
    assert r.accuracy[MEAN].value == pytest.approx(sum(map(sum, DEC)) / 16)
    assert r.unreadable_rate == 0.25
    assert r.n_q1_matched == 2 and r.q1_matched["latent_severity"] == 0.5 and r.q1_matched["effect"] is None
    assert r.cost_per_1000 == pytest.approx((1000 * 1.0 + 100 * 10.0) / 1e6 * 1000)
    assert r.latency_source == ROUND_TRIP and r.latency_p50 == pytest.approx(2.5)  # write_model: 1 + item position
    hosted = row_result(spec("Dec"), m, SELF_HOSTED, cumulative=False)
    assert hosted.cost_per_1000 is None and hosted.params_b == 4 and hosted.active_params_b is None


def test_row_result_stops_when_per_item_and_summary_disagree(tmp_path):
    m = write_run(tmp_path / "results", "dec_run", "dec-1", DEC, latent_risk_accuracy=0.1)
    with pytest.raises(ValueError, match="summary"):
        row_result(spec("Dec"), m, SELF_HOSTED, cumulative=False)


def system(name: str, correct: list[list[bool]]) -> System:
    return System(name, {i: ItemResult(tuple(c), None, 1.0) for i, c in zip(ITEMS, correct, strict=True)}, {})


def test_contrasts_use_paired_bootstrap_mcnemar_and_holm_over_the_whole_family():
    a, b, c = system("A", DEC), system("B", GEN), system("C", [[True] * 4] * 4)
    specs = [
        ContrastSpec(a="A", b="B", kind="pre-registered", changes="x"),
        ContrastSpec(a="C", b="B", kind="exploratory", changes="y"),
    ]
    results = contrast_family(specs, {"A": a, "B": b, "C": c})
    first = results[0]
    assert [t.metric for t in first.tests] == [MEAN, *QUESTIONS]
    mean = first.tests[0]
    a_mean = [sum(x) / 4 for x in DEC]
    b_mean = [sum(x) / 4 for x in GEN]
    assert (mean.diff.value, mean.diff.lo, mean.diff.hi) == paired_bootstrap_diff(a_mean, b_mean)
    assert mean.p is None and mean.p_holm is None
    q1 = first.tests[1]
    assert (q1.a_only, q1.b_only) == (2, 0)  # items 1, 2: A right, B wrong
    assert q1.p == mcnemar_exact(2, 0)
    raw = [t.p for r in results for t in r.tests[1:]]
    assert [t.p_holm for r in results for t in r.tests[1:]] == holm([p for p in raw if p is not None])


def test_contrasts_stop_when_item_sets_differ():
    a = system("A", DEC)
    b = System("B", {i: x for i, x in system("B", GEN).items.items() if i != "t4"}, {})
    with pytest.raises(ValueError, match="items"):
        contrast_family([ContrastSpec(a="A", b="B", kind="exploratory", changes="x")], {"A": a, "B": b})


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    results = tmp_path / "results"
    write_run(results, "dec_run", "dec-1", DEC)
    write_run(results, "gen_run", "gen-1", GEN, {"raw": {"remote": {"compute_s": 0.5}}})
    (results / "run_ids.yaml").write_text("dec_run: txt1\ngen_run: txt2\n")
    config = tmp_path / "same.yaml"
    config.write_text(yaml.safe_dump(cfg_dict()))
    prices = tmp_path / "prices.yaml"
    models = {"dec-1": {"input_per_mtok": 1.0, "output_per_mtok": 10.0}, "gen-1": {"self_hosted": "Modal L40S", "params_b": 4}}
    prices.write_text(yaml.safe_dump({"checked": "2026-10-12", "sources": {}, "models": models}))
    return {"results": results, "figures": tmp_path / "figures", "config": config, "prices": prices}


def args(tree: dict[str, Path]) -> list[str]:
    return [f"--{k}={v}" for k, v in tree.items()]


def test_script_writes_tables_and_figures_then_check_passes(tree):
    assert same_size_script.main(args(tree)) == 0
    out = tree["results"] / "same"
    assert {p.name for p in out.iterdir()} == {"tables.md", "rows.csv", "contrasts.csv"}
    figs = {p.name for p in (tree["figures"] / "same").iterdir()}
    assert {"accuracy_vs_latency.png", "accuracy_vs_unreadable.png"} <= figs
    tables = (out / "tables.md").read_text()
    assert "**EXPLORATORY**" in tables.splitlines()[:6]
    assert "high parse failures" in tables and "self-hosted, GPU time" in tables and "Modal compute_s" in tables
    assert "txt1" in tables and "pre-registered" in tables
    for name in ("rows.csv", "contrasts.csv"):
        assert (out / name).read_text().startswith("# EXPLORATORY")
    assert same_size_script.main([*args(tree), "--check"]) == 0
    (out / "rows.csv").write_text("edited\n")
    assert same_size_script.main([*args(tree), "--check"]) == 1


def test_script_stops_with_a_clear_error_when_a_configured_run_is_missing(tree):
    cfg = cfg_dict()
    cfg["rows"].append({"run": "part2_qwen_gen_fewshot", "model": "gen-1", "label": "Few", "group": "generate"})
    tree["config"].write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="part2_qwen_gen_fewshot"):
        same_size_script.main(args(tree))
