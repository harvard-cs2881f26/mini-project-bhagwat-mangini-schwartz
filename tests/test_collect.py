"""scripts/collect.py on a small synthetic results/ tree (text + video + a mock run that is skipped)."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from asimov.claims import Claims
from asimov.collect.build import build
from asimov.collect.load import load_results

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect  # noqa: E402  # pyright: ignore[reportMissingImports]

ITEMS = ["t1", "t2", "t3", "t4"]
QUESTIONS = ("latent_risk", "latent_severity", "effect", "activated_risk")


def text_outcome(correct: list[bool], pred: str = "A") -> dict:
    out: dict = {q: {"pred": pred if ok else "B", "gold": pred, "correct": ok} for q, ok in zip(QUESTIONS, correct, strict=True)}
    out["parse"] = dict.fromkeys(QUESTIONS, "ok")
    out["parse_ok"] = True
    out["finish"] = "complete"
    return out


def write_model(run_dir: Path, name: str, model: str, provider: str, benchmark: str, per_item: dict, metrics: dict) -> None:
    mdir = run_dir / model
    mdir.mkdir(parents=True)
    key = f"k-{model}"
    summary = {
        "name": name,
        "provider": provider,
        "model": model,
        "prompt": "p",
        "thinking": "min",
        "benchmark": benchmark,
        "run": run_dir.name,
        "key": key,
        "errors": 0,
        "input_tokens": 1000,
        "output_tokens": 100,
        "cached_input_tokens": 0,
        "truncation_rate": 0.0,
        "refusal_rate": 0.0,
        "n_with_reasoning": 0,
        "scored_at": "2026-10-12T10:00:00-0400",
        "git_commit": "abc1234def",
        "n": len(per_item),
        **metrics,
    }
    (mdir / "summary.json").write_text(json.dumps({**summary, "per_item": per_item}))
    rows = [
        {"id": i, "key": key, "served_model": f"{model}-2025", "latency_s": 1.0 + n, "text": f"reply {i}"}
        for n, i in enumerate(per_item)
    ]
    rows.append({"id": "t1", "key": "old-key", "served_model": "stale", "latency_s": 99.0, "text": "old"})
    (mdir / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def text_run(results: Path, run: str, large_correct: list[list[bool]]) -> None:
    run_dir = results / run
    models = [("GPT-5", "gpt-5", large_correct), ("GPT-5 Nano", "gpt-5-nano", [[False, True, False, True]] * 4)]
    cfg = {"run": run, "benchmark": "text", "models": [{"name": n, "provider": "openai", "model": m} for n, m, _ in models]}
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name, model, by_item in models:
        per_item = {i: text_outcome(c) for i, c in zip(ITEMS, by_item, strict=True)}
        metrics = {f"{q}_accuracy": sum(per_item[i][q]["correct"] for i in ITEMS) / 4 for q in QUESTIONS}
        metrics |= {"parse_failure_rate": 0.0, "latent_severity_n_severe": 2, "latent_severity_severe_recall": 0.5}
        write_model(run_dir, name, model, "openai", "text", per_item, metrics)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    res = tmp_path / "results"
    text_run(res, "text_run", [[True, True, False, True]] * 4)
    vdir = res / "video_run"
    video_items = {
        f"v{i}": {
            "case": "error",
            "truth_s": 2.0,
            "pred_s": 2.0 + i,
            "error_s": float(i),
            "within_0_5": i == 0,
            "risk_ok": i != 2,
            "finish": "complete",
        }
        for i in range(3)
    }
    write_model(
        vdir,
        "GPT-5",
        "gpt-5",
        "openai",
        "video",
        video_items,
        {"risk_accuracy": 2 / 3, "within_0_5_rate": 1 / 3, "timing_mae_s": 1.0, "parse_failure_rate": 0.0},
    )
    write_model(res / "smoke", "Mock", "mock-1", "mock", "text", {"t1": text_outcome([True] * 4)}, {})
    (res / "comparisons.yaml").write_text("comparisons: []\n")
    return res


@pytest.fixture
def claims() -> Claims:
    return Claims.model_validate(
        {
            "paper": {"arxiv": "2509.21651", "version": "v2", "title": "T", "read_on": "2026-10-12"},
            "claims": [
                {
                    "id": "fig4a_gpt5_latent_risk",
                    "figure": "4a",
                    "section": "4",
                    "benchmark": "text",
                    "model": "GPT 5",
                    "metric": "latent_risk_accuracy",
                    "value": 0.9,
                    "scale": "fraction",
                    "kind": "value",
                    "source": "figure",
                    "quote": "read off figure",
                    "approx": True,
                },
                {
                    "id": "fig4a_opus_latent_risk",
                    "figure": "4a",
                    "section": "4",
                    "benchmark": "text",
                    "model": "Claude Opus 4.1",
                    "metric": "latent_risk_accuracy",
                    "value": 0.95,
                    "scale": "fraction",
                    "kind": "value",
                    "source": "figure",
                    "quote": "read off figure",
                    "approx": True,
                },
                {
                    "id": "fig5b_gpt5_within",
                    "figure": "5b",
                    "section": "5",
                    "benchmark": "video",
                    "model": "GPT 5",
                    "metric": "within_0_5_rate",
                    "value": 40.0,
                    "scale": "percent",
                    "kind": "value",
                    "source": "figure",
                    "quote": "read off figure",
                    "approx": True,
                },
            ],
            "headline": [
                {"id": "a", "check": "close_to_paper", "section": "4", "quote": "q", "models": ["GPT 5"]},
                {"id": "b", "check": "size_gap", "section": "4", "quote": "q", "pairs": [["GPT 5", "GPT 5 Nano"]]},
                {"id": "c", "check": "actions_harder", "section": "4", "quote": "q"},
            ],
        }
    )


def read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def test_index_has_one_row_per_run_and_model_and_skips_mock_runs(results, claims):
    index = read_csv(build(load_results(results), claims, {}, {}, [])["index.csv"])
    assert [(r["run"], r["model"]) for r in index] == [("text_run", "gpt-5"), ("text_run", "gpt-5-nano"), ("video_run", "gpt-5")]
    first = index[0]
    assert first["served_model"] == "gpt-5-2025"  # rows under an old key are ignored
    assert float(first["latency_p50_s"]) == pytest.approx(2.5)
    assert first["cost_usd"] == ""  # no price table
    assert first["latent_risk_accuracy"] == "1.0"


def test_cost_from_price_table(results, claims):
    prices = {"gpt-5": {"input_per_mtok": 1.0, "output_per_mtok": 10.0}}
    index = read_csv(build(load_results(results), claims, prices, {}, [])["index.csv"])
    assert float(index[0]["cost_usd"]) == pytest.approx(1000 / 1e6 + 100 * 10 / 1e6)
    assert index[1]["cost_usd"] == ""


def test_per_item_rows(results, claims):
    rows = read_csv(build(load_results(results), claims, {}, {}, [])["per_item.csv"])
    assert len(rows) == 2 * 4 * 4 + 3 * 2
    video = [r for r in rows if r["benchmark"] == "video"]
    assert {r["question"] for r in video} == {"risk", "timing"}
    assert [r["correct"] for r in video if r["question"] == "risk"] == ["True", "True", "False"]


def test_pages_exist_and_are_deterministic(results, claims):
    runs = load_results(results)
    first = build(runs, claims, {}, {}, [])
    assert first == build(load_results(results), claims, {}, {}, [])
    for path in ["README.md", "text_run/README.md", "text_run/paper_check.md", "text_run/table.csv", "video_run/README.md"]:
        assert path in first
    assert "smoke/README.md" not in first


def test_paper_check(results, claims):
    page = build(load_results(results), claims, {}, {}, [])["text_run/paper_check.md"]
    assert "fig4a_gpt5_latent_risk" in page
    assert "not in this run" in page  # Claude Opus 4.1 wasn't run
    table = read_csv(build(load_results(results), claims, {}, {}, [])["text_run/table.csv"])
    gpt5 = next(r for r in table if r["claim"] == "fig4a_gpt5_latent_risk")
    assert float(gpt5["ours"]) == 1.0 and float(gpt5["diff"]) == pytest.approx(0.1)
    assert gpt5["paper_in_ci"] == "True"  # Wilson CI for 4/4 reaches down to 0.51
    video = read_csv(build(load_results(results), claims, {}, {}, [])["video_run/table.csv"])
    assert float(video[0]["paper"]) == pytest.approx(0.4)  # percent claims compared as fractions


def test_headline_checks_on_text_run(results, claims):
    page = build(load_results(results), claims, {}, {}, [])["text_run/paper_check.md"]
    assert "Holm" in page  # size gap: GPT 5 minus GPT 5 Nano, per question
    assert "GPT-5 minus GPT-5 Nano" in page
    assert "effect + activated_risk" in page


def test_comparison_page(results, claims, tmp_path):
    text_run(results, "text_run_b", [[False, False, False, False]] * 4)
    comparisons = [{"name": "a_vs_b", "a": "text_run", "b": "text_run_b"}]
    files = build(load_results(results), claims, {}, {}, comparisons)
    page = files["compare/a_vs_b.md"]
    assert "McNemar" in page and "<details>" in page


def test_cli_writes_then_check_passes_then_fails_after_edit(results, tmp_path):
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(
        yaml.safe_dump(
            {"paper": {"arxiv": "x", "version": "v2", "title": "T", "read_on": "2026-10-12"}, "claims": [], "headline": []}
        )
    )
    args = ["--results", str(results), "--claims", str(claims_path)]
    assert collect.main(args) == 0
    assert (results / "index.csv").exists() and (results / "headline.yaml").exists()
    assert collect.main([*args, "--check"]) == 0
    (results / "text_run" / "README.md").write_text("edited by hand\n")
    assert collect.main([*args, "--check"]) == 1


def test_headline_yaml_is_never_overwritten(results, tmp_path):
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(
        yaml.safe_dump(
            {"paper": {"arxiv": "x", "version": "v2", "title": "T", "read_on": "2026-10-12"}, "claims": [], "headline": []}
        )
    )
    (results / "headline.yaml").write_text("figures:\n  fig4: {run: text_run, note: mine}\n")
    collect.main(["--results", str(results), "--claims", str(claims_path)])
    assert "note: mine" in (results / "headline.yaml").read_text()


def test_collect_never_imports_the_data_loaders(results, tmp_path):
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(
        yaml.safe_dump(
            {"paper": {"arxiv": "x", "version": "v2", "title": "T", "read_on": "2026-10-12"}, "claims": [], "headline": []}
        )
    )
    code = (
        "import runpy, sys\n"
        "sys.argv = ['collect.py', '--results', sys.argv[1], '--claims', sys.argv[2]]\n"
        "try:\n    runpy.run_path('scripts/collect.py', run_name='__main__')\nexcept SystemExit:\n    pass\n"
        "print(sorted(m for m in sys.modules if m.startswith('asimov.data')))"
    )
    env = {"ASIMOV_DATA_DIR": str(tmp_path / "missing")}
    out = subprocess.run(
        [sys.executable, "-c", code, str(results), str(claims_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"
