"""scripts/collect.py on a small synthetic results/ tree (text + video + a mock run that is skipped)."""

import csv
import io
import json
import re
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
from asimov.collect.pages import FIGURE_SUBHEADINGS, FIGURE_VERSION_NOTES, FIGURES, GENERATED, FailureCounts, failure_counts
from asimov.collect.run_ids import assign_ids, describe
from asimov.collect.tables import CSV_DIGITS, to_csv
from asimov.run_ids import PREFIXES, next_number, parse_ids

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect  # noqa: E402  # pyright: ignore[reportMissingImports]
from results_tree import make_results, text_run, write_model  # noqa: E402  # pyright: ignore[reportMissingImports]


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


# ---- run IDs ----------------------------------------------------------------------------------


def write_claims(tmp_path: Path) -> Path:
    path = tmp_path / "claims.yaml"
    path.write_text(yaml.safe_dump(empty_claims().model_dump()))
    return path


def test_prefixes_cover_every_benchmark():
    assert PREFIXES == {"text": "txt", "video": "vid", "image": "img", "image_rationale": "imgr"}


def test_new_runs_get_the_next_number_per_prefix_by_scored_at(results):
    text_run(results, "text_early", [[True] * 4] * 4, scored_at="2026-10-11T09:00:00-0400")
    text_run(results, "text_late", [[True] * 4] * 4, scored_at="2026-10-13T09:00:00-0400")
    ids = assign_ids({}, load_results(results))
    # text_run is scored 2026-10-12, between the other two; ties would go by name
    assert ids == {"text_early": "txt0", "text_run": "txt1", "text_late": "txt2", "video_run": "vid0"}


def test_existing_ids_are_never_changed(results):
    text_run(results, "text_early", [[True] * 4] * 4, scored_at="2026-10-11T09:00:00-0400")
    existing = {"text_run": "txt0", "video_run": "vid0"}
    ids = assign_ids(existing, load_results(results))
    assert ids == {"text_run": "txt0", "video_run": "vid0", "text_early": "txt1"}  # earlier, but text_run keeps txt0
    assert existing == {"text_run": "txt0", "video_run": "vid0"}  # input not mutated


def test_a_removed_runs_number_is_never_reused(results):
    existing = {"deleted_run": "txt0", "old_image": "imgr3"}
    ids = assign_ids(existing, load_results(results))
    assert ids == {"deleted_run": "txt0", "old_image": "imgr3", "text_run": "txt1", "video_run": "vid0"}


def test_img_and_imgr_numbers_are_separate():
    assert assign_ids({"a": "imgr0", "b": "imgr1"}, []) == {"a": "imgr0", "b": "imgr1"}
    assert next_number({"a": "imgr0", "b": "imgr1"}, "img") == 0
    assert next_number({"a": "img4", "b": "imgr7"}, "img") == 5


def test_run_id_column_follows_run_in_both_tables(results):
    files = build(load_results(results), empty_claims(), {}, {}, [], {"text_run": "txt0", "video_run": "vid0"})
    index = read_csv(files["index.csv"])
    assert list(index[0])[:3] == ["run", "run_id", "description"]
    assert [r["run_id"] for r in index] == ["txt0", "txt0", "vid0"]
    assert index[0]["description"] == (
        "Text benchmark, using the p prompt, with thinking set to min (the lowest reasoning each provider allows), on 2 models."
    )
    assert index[2]["description"].startswith("Video benchmark,") and index[2]["description"].endswith(
        "on 1 model."
    )  # no config.yaml: from summaries
    per_item = read_csv(files["per_item.csv"])
    assert list(per_item[0])[:3] == ["run", "run_id", "benchmark"] and "description" not in per_item[0]
    assert {r["run_id"] for r in per_item if r["run"] == "video_run"} == {"vid0"}


def test_run_id_and_description_appear_on_the_pages(results):
    text_run(results, "text_run_b", [[False] * 4] * 4, thinking="default")
    ids = {"text_run": "txt0", "text_run_b": "txt1", "video_run": "vid0"}
    headline = {"figures": {"fig4": {"run": "text_run", "note": ""}}}
    comparisons = [{"name": "a_vs_b", "a": "text_run", "b": "text_run_b"}]
    files = build(load_results(results), empty_claims(), {}, headline, comparisons, ids)
    first = (
        "Text benchmark, using the p prompt, with thinking set to min (the lowest reasoning each provider allows), on 2 models."
    )
    assert f"# Run `txt0`: `text_run`\n\n{first}\n" in files["text_run/README.md"]
    later = (
        "Text benchmark, using the p prompt, with thinking set to default (each provider decides how much to reason), "
        "on 2 models."
    )
    assert later in files["text_run_b/README.md"]
    summary = files["README.md"]
    assert "| Run ID | Run | Description |" in summary
    assert f"| `txt1` | [`text_run_b`](text_run_b/README.md) | {later} |" in summary
    assert "Run: `txt0` [`text_run`](text_run/README.md)" in summary
    compare = files["compare/a_vs_b.md"]
    assert "`txt0`" in compare.splitlines()[2] and "`txt1`" in compare.splitlines()[2]
    assert f"A: {first}" in compare and f"B: {later}" in compare
    assert "(auto)" not in summary


def describe_all(results: Path) -> dict[str, str]:
    return {r.name: describe(r) for r in load_results(results)}


def test_description_states_the_runs_own_settings(results):
    text_run(results, "verbatim", [[True] * 4] * 4, prompt="text_verbatim", log_reasoning=True)
    assert describe_all(results)["verbatim"] == (
        "Text benchmark, using the paper's prompt sent unchanged (text_verbatim), with thinking set to min "
        "(the lowest reasoning each provider allows), on 2 models. Reasoning summaries are saved."
    )


def test_thinking_default_and_other_values_read_as_words(results):
    text_run(results, "text_default", [[True] * 4] * 4, thinking="default")
    text_run(results, "text_high", [[True] * 4] * 4, thinking="high")
    described = describe_all(results)
    assert "with thinking set to default (each provider decides how much to reason)," in described["text_default"]
    assert "with thinking set to high," in described["text_high"]
    assert described["video_run"].startswith("Video benchmark, using the p prompt,")


def test_description_never_refers_to_other_runs_or_order(results):
    text_run(results, "text_b", [[True] * 4] * 4, extra_models=("o3",), thinking="default", prompt="q")
    for text in describe_all(results).values():
        for word in ("first", "same as", "rerun", "instead of", "except"):
            assert word not in text.lower(), (word, text)
        assert not re.search(r"\b(txt|vid|img)\d", text), text
    assert describe_all(results)["text_b"].endswith("on 3 models.")


def test_descriptions_are_deterministic(results):
    assert describe_all(results) == describe_all(results)


def test_run_ids_yaml_rejects_old_style_entries():
    assert parse_ids({"a": "txt0", "b": "imgr2"}) == {"a": "txt0", "b": "imgr2"}
    with pytest.raises(ValueError, match="run name -> id"):
        parse_ids({"a": {"id": "txt0", "description": ""}})
    with pytest.raises(ValueError):
        parse_ids({"a": "not an id"})


def test_cli_writes_run_ids_and_check_fails_when_one_is_missing(results, tmp_path):
    args = ["--results", str(results), "--claims", str(write_claims(tmp_path))]
    assert collect.main(args) == 0
    path = results / "run_ids.yaml"
    text = path.read_text()
    assert text.startswith("#") and "never changed or reused" in text and "description" not in text
    assert yaml.safe_load(text) == {"text_run": "txt0", "video_run": "vid0"}
    assert collect.main([*args, "--check"]) == 0

    text_run(results, "text_run_b", [[False] * 4] * 4, scored_at="2026-10-01T09:00:00-0400")
    assert collect.main(args) == 0  # rebuild pages so only run_ids.yaml can be stale below
    assert yaml.safe_load(path.read_text())["text_run_b"] == "txt1"  # new run: next number, even though scored earlier
    path.write_text(text)  # drop text_run_b's ID
    before = {p: p.read_text() for p in results.rglob("*") if p.is_file()}
    assert collect.main([*args, "--check"]) == 1
    assert {p: p.read_text() for p in results.rglob("*") if p.is_file()} == before  # --check writes nothing
    assert collect.main(args) == 0
    assert collect.main([*args, "--check"]) == 0


# ---- run page as a report --------------------------------------------------------------------

RUN_PAGE_SECTIONS = [
    "# Run `txt0`: `text_run`",
    "## Write-up",
    "## Figures",
    "## Comparison with the paper",
    "## Models",
    "## Metrics (95% CI)",
    "## Files",
]
IDS = {"text_run": "txt0", "video_run": "vid0"}
HEADLINE = {"figures": {"fig4": {"run": "text_run"}, "fig5": {"run": "video_run"}, "fig6": {"run": None}}}


def run_page(results: Path, claims: Claims, run: str = "text_run", headline: dict | None = None) -> str:
    return build(load_results(results), claims, {}, HEADLINE if headline is None else headline, [], IDS)[f"{run}/README.md"]


def test_run_page_has_its_sections_in_order(results, claims):
    page = run_page(results, claims)
    positions = [page.index(s + "\n") for s in RUN_PAGE_SECTIONS]
    assert positions == sorted(positions)
    assert page.index("Text benchmark,") < page.index("## Write-up")  # description right under the title


def test_run_page_folds_in_the_paper_check(results, claims):
    page = run_page(results, claims)
    comparison = page[page.index("## Comparison with the paper") : page.index("## Models")]
    assert "fig4a_gpt5_latent_risk" in comparison and "[paper_check.md](paper_check.md)" in comparison
    assert "Headline claims" in comparison and "Holm" in comparison and "effect + activated_risk" in comparison
    assert "fig5b_gpt5_within" in run_page(results, claims, "video_run")


def test_run_page_lists_the_runs_figures_comparison_first(results, claims):
    page = run_page(results, claims)
    figures = page[page.index("## Figures") : page.index("## Comparison with the paper")]
    links = re.findall(r"\]\((\.\./\.\./figures/[^)]+)\)", figures)
    assert links == [f"../../figures/txt0/{f}_{v}.png" for v in ("comparison", "paper") for f in ("fig4a", "fig4b", "fig5a")] + [
        "../../figures/txt0/severity_scoring.png"  # text runs' extra chart, after both versions
    ]
    video = run_page(results, claims, "video_run")
    assert re.findall(r"\]\(\.\./\.\./figures/vid0/([^)]+)\)", video) == [
        "fig5a_comparison.png",
        "fig5b_comparison.png",
        "fig5a_paper.png",
        "fig5b_paper.png",
    ]


def test_fig5a_is_left_out_without_a_headline_run_of_the_other_benchmark(results, claims):
    page = run_page(results, claims, headline={"figures": {"fig4": {"run": "text_run"}, "fig5": {"run": None}}})
    assert "fig4a_comparison.png" in page and "fig5a" not in page


def test_writeup_is_included_verbatim_when_present(results, claims):
    assert "No write-up yet: add results/text_run/writeup.md" in run_page(results, claims)
    writeup = "### What we see\n\nGPT-5 | table pipes and *markdown* stay as written.\n"
    (results / "text_run" / "writeup.md").write_text(writeup)
    page = run_page(results, claims)
    assert writeup.strip() in page and "No write-up yet" not in page
    assert page.index("## Write-up") < page.index(writeup.strip()) < page.index("## Figures")


def test_collect_never_writes_writeup(results, tmp_path):
    args = ["--results", str(results), "--claims", str(write_claims(tmp_path))]
    assert collect.main(args) == 0
    assert not (results / "text_run" / "writeup.md").exists()  # no placeholder file
    path = results / "video_run" / "writeup.md"
    path.write_text("Hand-written.\n")
    files = build(load_results(results), empty_claims(), {}, {}, [])
    assert not any(p.endswith("writeup.md") for p in files)
    assert collect.orphaned_files(results, files) == []
    assert collect.main([*args, "--check"]) == 1  # the run page includes the write-up: rerun collect
    assert collect.main(args) == 0 and path.read_text() == "Hand-written.\n"
    assert collect.main([*args, "--check"]) == 0
    assert "Hand-written." in (results / "video_run" / "README.md").read_text()


def test_summary_links_the_headline_runs_figures(results, claims):
    summary = build(load_results(results), claims, {}, HEADLINE, [], IDS)["README.md"]
    assert "![fig4a](../figures/txt0/fig4a_comparison.png)" in summary
    assert "![fig4b](../figures/txt0/fig4b_comparison.png)" in summary
    assert "![fig5a](../figures/vid0/fig5a_comparison.png)" in summary
    assert "![fig5b](../figures/vid0/fig5b_comparison.png)" in summary
    assert "](../figures/fig" not in summary  # no old top-level figure paths


# ---- run folders: results/<id>_<run>/ -----------------------------------------------------------


def id_folder_results(tmp_path: Path) -> Path:
    """Runs in results/<id>_<run>/ folders, a comparison and headline runs, as the runner now writes them."""
    res = tmp_path / "results"
    text_run(res, "repro_a", [[True, True, False, True]] * 4, folder="txt0_repro_a")
    text_run(res, "repro_b", [[False, False, False, True]] * 4, folder="txt1_repro_b")
    video = {"v0": {"case": "error", "truth_s": 2.0, "pred_s": 2.0, "error_s": 0.0, "within_0_5": True, "risk_ok": True}}
    write_model(res / "vid0_vid_a", "GPT-5", "gpt-5", "openai", "video", video, {"risk_accuracy": 1.0}, run="vid_a")
    (res / "run_ids.yaml").write_text(yaml.safe_dump({"repro_a": "txt0", "repro_b": "txt1", "vid_a": "vid0"}))
    (res / "comparisons.yaml").write_text(yaml.safe_dump({"comparisons": [{"name": "a_vs_b", "a": "repro_a", "b": "repro_b"}]}))
    figures = {"fig4": {"run": "repro_a", "note": ""}, "fig5": {"run": "vid_a", "note": ""}}
    (res / "headline.yaml").write_text(yaml.safe_dump({"figures": figures}))
    return res


def test_run_name_comes_from_the_config_not_the_folder(tmp_path):
    res = tmp_path / "results"
    text_run(res, "logical_name", [[True] * 4] * 4, folder="txt9_something_else")
    (run,) = load_results(res)
    assert (run.name, run.folder) == ("logical_name", "txt9_something_else")
    files = build([run], empty_claims(), {}, {}, [], {"logical_name": "txt9"})
    assert "txt9_something_else/README.md" in files and "# Run `txt9`: `logical_name`" in files["txt9_something_else/README.md"]


def test_run_name_falls_back_to_the_summaries(tmp_path):
    res = tmp_path / "results"
    text_run(res, "from_summaries", [[True] * 4] * 4, folder="txt0_from_summaries")
    (res / "txt0_from_summaries" / "config.yaml").unlink()
    assert [r.name for r in load_results(res)] == ["from_summaries"]


def test_page_links_point_to_existing_files(tmp_path):
    res = id_folder_results(tmp_path)
    args = ["--results", str(res), "--claims", str(write_claims(tmp_path))]
    assert collect.main(args) == 0 and collect.main([*args, "--check"]) == 0
    files = build(load_results(res), empty_claims(), {}, load_yaml_file(res / "headline.yaml"), load_comparisons(res))
    assert {"txt0_repro_a/README.md", "txt1_repro_b/README.md", "vid0_vid_a/README.md", "compare/a_vs_b.md"} <= set(files)
    checked = 0
    for rel in (p for p in files if p.endswith(".md")):
        page = res / rel
        for link in re.findall(r"\]\(([^)#\s]+)\)", page.read_text()):
            target = (page.parent / link).resolve()
            if link.startswith("http") or not target.is_relative_to(res.resolve()):
                continue  # figures/ and WORKLOG.md sit outside this temporary results/
            assert target.exists(), f"{rel}: broken link {link}"
            checked += 1
    assert checked > 10
    summary = (res / "README.md").read_text()
    assert "(txt0_repro_a/README.md)" in summary and "(vid0_vid_a/README.md)" in summary


def load_yaml_file(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_comparisons(res: Path) -> list[dict]:
    return load_yaml_file(res / "comparisons.yaml")["comparisons"]


def test_each_figure_subheading_has_its_plain_language_note(results, claims):
    page = run_page(results, claims)
    for version, heading in FIGURE_SUBHEADINGS.items():
        note = FIGURE_VERSION_NOTES[version]
        assert f"### {heading}\n\n{note}\n\n![" in page
    assert page.index(FIGURE_VERSION_NOTES["comparison"]) < page.index(FIGURE_VERSION_NOTES["paper"])


def test_mock_runs_get_no_id_and_keep_a_plain_folder(tmp_path):
    from asimov.run_ids import run_dir

    assert run_dir(tmp_path, "smoke_mock", "video", ["mock"]) == tmp_path / "smoke_mock"
    assert not (tmp_path / "run_ids.yaml").exists()
    assert run_dir(tmp_path, "real_run", "video", ["openai", "mock"]).name == "vid0_real_run"


def test_text_run_page_shows_the_scoring_chart(results):
    from asimov.collect.pages import EXTRA_SUBHEADING

    files = build(load_results(results), empty_claims(), {}, {}, [], {"text_run": "txt0", "video_run": "vid0"})
    text_page = next(v for k, v in files.items() if k.endswith("README.md") and "text_run" in k)
    assert f"### {EXTRA_SUBHEADING}" in text_page and "figures/txt0/severity_scoring.png" in text_page
    video_page = next(v for k, v in files.items() if k.endswith("README.md") and "video_run" in k)
    assert EXTRA_SUBHEADING not in video_page
