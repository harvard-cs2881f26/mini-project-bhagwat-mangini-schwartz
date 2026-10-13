"""Runner tests with mock models: no keys, no network."""
import json

import pytest
import yaml
from pydantic import ValidationError

from asimov.bench.video_score import score
from asimov.data.video import load_video_answers
from asimov.models.mock import REPLIES
from asimov.runner import ConfigChanged, run

CFG = {"run": "t", "benchmark": "video", "prompt": "video_v2", "thinking": "min", "models": [
    {"name": "Always No", "provider": "mock", "model": "mock-always-no"},
    {"name": "Broken", "provider": "mock", "model": "mock-error"},
]}


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def test_run_scores_saves_and_reuses(tmp_path):
    # small limits so this also runs on the 5-item sample in tests/fixtures/data
    rows = run(CFG, limit=2, results_dir=tmp_path)
    no, broken = rows
    saved = lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")
    assert len(saved) == 2
    ids = [json.loads(line)["id"] for line in saved]
    expected = score(load_video_answers(), {i: REPLIES["mock-always-no"] for i in ids})
    assert no["risk_accuracy"] == expected["risk_accuracy"] and no["n"] == 2
    # failed calls are reported, never saved
    assert broken["errors"] == 2 and broken["n"] == 0
    assert lines(tmp_path / "t" / "mock-error" / "responses.jsonl") == []
    # rerun: nothing new is called or written; growing the limit only adds the new videos
    run(CFG, limit=2, results_dir=tmp_path)
    assert len(lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")) == 2
    run(CFG, limit=4, results_dir=tmp_path)
    assert len(lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")) == 4


def test_limit_order_is_mixed_and_fixed(tmp_path):
    from asimov.data.video import load_video_items
    from asimov.runner import fixed_order
    first = [i.id for i in fixed_order(load_video_items())[:20]]
    assert first == [i.id for i in fixed_order(load_video_items())[:20]]
    answers = load_video_answers()
    assert {answers[i].risk for i in first} == {"Yes", "No"}


def test_unknown_model_warns(tmp_path, capsys):
    assert run(CFG, limit=1, only=["gemini-3.1-flash-lite"], results_dir=tmp_path) == []
    assert "not in this config" in capsys.readouterr().out


# --- run provenance and the one-config-per-run guard (docs/PLAN.md Session 1) ---

NEW_FIELDS = {"benchmark", "run", "key", "config_path", "config_sha256", "git_commit", "git_dirty", "scored_at", "n"}


def test_summary_records_how_it_was_made(tmp_path):
    no = run(CFG, limit=2, results_dir=tmp_path)[0]
    saved = json.loads((tmp_path / "t" / "mock-always-no" / "summary.json").read_text())
    for summary in (no, saved):
        assert NEW_FIELDS <= set(summary)
        assert summary["benchmark"] == "video" and summary["run"] == "t" and summary["n"] == 2
        assert len(summary["config_sha256"]) == 64
    assert saved["key"] == json.loads(lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")[0])["key"]


def test_config_copy_is_written_and_guards_the_run_name(tmp_path):
    run(CFG, limit=1, results_dir=tmp_path)
    copy = tmp_path / "t" / "config.yaml"
    assert yaml.safe_load(copy.read_text()) == CFG
    run(CFG, limit=1, results_dir=tmp_path)  # same config: fine
    changed = {**CFG, "thinking": "default"}
    with pytest.raises(ConfigChanged, match="new `run:` name"):
        run(changed, limit=1, results_dir=tmp_path)
    assert yaml.safe_load(copy.read_text()) == CFG  # untouched


def test_config_path_is_copied_byte_for_byte(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("# a comment\n" + yaml.safe_dump(CFG))
    run(CFG, limit=1, results_dir=tmp_path / "res", config_path=path)
    assert (tmp_path / "res" / "t" / "config.yaml").read_bytes() == path.read_bytes()


def test_unknown_config_key_fails_before_any_call(tmp_path):
    with pytest.raises(ValidationError):
        run({**CFG, "temperature": 0}, limit=1, results_dir=tmp_path)
    assert not (tmp_path / "t").exists()


def test_no_calls_scores_saved_replies_only(tmp_path):
    run(CFG, limit=2, results_dir=tmp_path)
    no, broken = run(CFG, limit=4, results_dir=tmp_path, call_models=False)
    assert no["n"] == 2  # the 2 saved replies, nothing new called
    assert len(lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")) == 2
    assert broken["n"] == 0 and lines(tmp_path / "t" / "mock-error" / "responses.jsonl") == []


def test_no_calls_keeps_the_last_error_count(tmp_path):
    # rescoring makes no calls, so it must not overwrite the failures the last real run recorded
    run(CFG, limit=2, results_dir=tmp_path)
    broken = run(CFG, limit=2, results_dir=tmp_path, call_models=False)[1]
    assert broken["errors"] == 2


def test_bad_prompt_or_provider_fails_before_the_config_copy(tmp_path):
    with pytest.raises(FileNotFoundError):  # video prompts are files: src/asimov/prompts/<name>.txt
        run({**CFG, "prompt": "video_v9"}, limit=1, results_dir=tmp_path)
    with pytest.raises(ValidationError):
        run({**CFG, "models": [{"name": "X", "provider": "opnai", "model": "gpt-5"}]}, limit=1, results_dir=tmp_path)
    assert not (tmp_path / "t" / "config.yaml").exists()


def test_comment_only_edit_keeps_the_run_name(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(CFG))
    run(CFG, limit=1, results_dir=tmp_path / "res", config_path=path)
    path.write_text("# a new comment\n" + yaml.safe_dump(CFG))
    run(CFG, limit=1, results_dir=tmp_path / "res", config_path=path)  # no ConfigChanged
    assert (tmp_path / "res" / "t" / "config.yaml").read_bytes() == path.read_bytes()


def test_no_calls_keeps_error_counts_after_a_partial_run(tmp_path):
    run(CFG, limit=2, results_dir=tmp_path)
    run(CFG, limit=2, results_dir=tmp_path, only=["mock-always-no"])  # run-level summary now lacks Broken
    broken = run(CFG, limit=2, results_dir=tmp_path, call_models=False)[1]
    assert broken["errors"] == 2
    assert json.loads((tmp_path / "t" / "mock-error" / "summary.json").read_text())["n"] == 0


# --- failed-call log, finish categories, cached tokens ---


def test_failed_calls_are_logged_but_never_saved_as_replies(tmp_path):
    run(CFG, limit=2, results_dir=tmp_path)
    errs = [json.loads(line) for line in lines(tmp_path / "t" / "mock-error" / "errors.jsonl")]
    assert len(errs) == 2 and {e["error_type"] for e in errs} == {"RuntimeError"}
    assert set(errs[0]) >= {"id", "key", "error_type", "message", "time"}
    assert lines(tmp_path / "t" / "mock-error" / "responses.jsonl") == []
    run(CFG, limit=2, results_dir=tmp_path)  # failures are retried, not reused from the log
    assert len(lines(tmp_path / "t" / "mock-error" / "errors.jsonl")) == 4


def test_finish_categories_and_cached_tokens_in_summary(tmp_path):
    no = run(CFG, limit=2, results_dir=tmp_path)[0]
    assert no["truncation_rate"] == 0 and no["refusal_rate"] == 0
    assert "cached_input_tokens" in no
    summary = json.loads((tmp_path / "t" / "mock-always-no" / "summary.json").read_text())
    assert {o["finish"] for o in summary["per_item"].values()} == {"complete"}
    row = json.loads(lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")[0])
    assert "cached_input_tokens" in row["usage"]


def test_subset_run_keeps_other_models_in_combined_summary(tmp_path):
    run(CFG, limit=2, only=["mock-always-no"], results_dir=tmp_path)
    run(CFG, limit=2, only=["mock-error"], results_dir=tmp_path)
    combined = json.loads((tmp_path / "t" / "summary.json").read_text())
    assert [r["model"] for r in combined] == ["mock-always-no", "mock-error"]
    assert combined[0]["n"] == 2 and "per_item" not in combined[0]
