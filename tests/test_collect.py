"""scripts/collect.py on a small synthetic results/ tree (text + video + a mock run that is skipped)."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from asimov.bench.text_score import QUESTIONS
from asimov.claims import Claims
from asimov.collect.build import build
from asimov.collect.checks import ACTION_QUESTIONS, LATENT_QUESTIONS
from asimov.collect.load import ModelResult, load_model, load_results
from asimov.collect.pages import FIGURES, GENERATED, FailureCounts, failure_counts
from asimov.collect.tables import CSV_DIGITS, to_csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect  # noqa: E402  # pyright: ignore[reportMissingImports]
from results_tree import make_results, text_run  # noqa: E402  # pyright: ignore[reportMissingImports]


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return make_results(tmp_path)


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


def model_result(per_item: dict, **summary) -> ModelResult:
    return ModelResult({"name": "M", "model": "m", "n": len(per_item), **summary}, per_item, (), (), {})


def test_failure_counts_from_per_item():
    per_item = {
        "a": {"parse_ok": False, "finish": "complete"},  # malformed
        "b": {"parse_ok": True, "finish": "truncated"},  # cut off but readable: not a parse failure
        "c": {"parse_ok": False, "finish": "truncated"},
        "d": {"parse_ok": False, "finish": "refused"},
        "e": {"parse_ok": True, "finish": "complete"},
    }
    m = model_result(per_item, parse_failure_rate=0.6, refusal_rate=0.2, truncation_rate=0.4)
    assert failure_counts(m) == FailureCounts(failed=3, refused=1, truncated=1, malformed=1)


def test_failure_counts_fall_back_to_summary_rates_without_parse_ok():
    per_item = {f"v{i}": {"finish": "refused" if i == 0 else "complete"} for i in range(4)}  # video: no parse_ok
    m = model_result(per_item, parse_failure_rate=0.5, refusal_rate=0.25, truncation_rate=0.0)
    assert failure_counts(m) == FailureCounts(failed=2, refused=1, truncated=0, malformed=None)


def test_csv_floats_are_rounded(results):
    assert to_csv([{"x": 2 / 3, "y": 1, "z": True}], ()) == f"x,y,z\n{round(2 / 3, CSV_DIGITS)},1,True\n"
    index = read_csv(build(load_results(results), empty_claims(), {}, {}, [])["index.csv"])
    assert index[-1]["risk_accuracy"] == "0.666667"


def empty_claims() -> Claims:
    return Claims.model_validate(
        {"paper": {"arxiv": "x", "version": "v2", "title": "T", "read_on": "2026-10-12"}, "claims": [], "headline": []}
    )


def test_question_groups_come_from_text_score():
    assert LATENT_QUESTIONS + ACTION_QUESTIONS == QUESTIONS
    assert FIGURES["fig4"][1] == tuple(f"{q}_accuracy" for q in QUESTIONS)


def test_rows_without_latency_are_skipped(results):
    path = results / "text_run" / "gpt-5" / "responses.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del rows[0]["latency_s"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = load_model(path.parent)
    assert len(m.latencies) == 3 and 0.0 not in m.latencies
    assert len(m.replies) == 4  # the row still counts for served model and reply text


def test_check_flags_orphaned_generated_files(results, tmp_path):
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(yaml.safe_dump(empty_claims().model_dump()))
    args = ["--results", str(results), "--claims", str(claims_path)]
    assert collect.main(args) == 0
    (results / "compare").mkdir()
    (results / "compare" / "gone.md").write_text(GENERATED + "\n")
    (results / "old_run").mkdir()
    (results / "old_run" / "README.md").write_text(GENERATED + "\n")
    (results / "old_run" / "paper_check.md").write_text(GENERATED + "\n")
    (results / "old_run" / "table.csv").write_text("claim\n")
    (results / "verify").mkdir()
    (results / "verify" / "README.md").write_text("# Hand-written\n")  # not generated: left alone
    files = build(load_results(results), empty_claims(), {}, {}, [])
    assert collect.orphaned_files(results, files) == [
        "compare/gone.md",
        "old_run/README.md",
        "old_run/paper_check.md",
        "old_run/table.csv",
    ]
    assert collect.main([*args, "--check"]) == 1
