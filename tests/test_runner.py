"""Runner tests with mock models: no keys, no network."""
import json

import pytest

from asimov.bench.video_score import score
from asimov.data.video import load_video_answers
from asimov.models.mock import REPLIES
from asimov.runner import run

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
    ids = [json.loads(l)["id"] for l in saved]
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
