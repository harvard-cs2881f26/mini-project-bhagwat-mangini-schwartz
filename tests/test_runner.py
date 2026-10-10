"""Runner tests with mock models: no keys, no network."""
import json

import pytest

from asimov.bench.video_score import score
from asimov.data.video import DATA_DIR, load_video_answers
from asimov.models.mock import REPLIES
from asimov.runner import run

pytestmark = pytest.mark.skipif(not (DATA_DIR / "video" / "items.jsonl").exists(), reason="data not downloaded")
CFG = {"run": "t", "prompt": "video_v1", "thinking": "min", "models": [
    {"name": "Always No", "provider": "mock", "model": "mock-always-no"},
    {"name": "Broken", "provider": "mock", "model": "mock-error"},
]}


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def test_run_scores_saves_and_reuses(tmp_path):
    rows = run(CFG, limit=10, results_dir=tmp_path)
    no, broken = rows
    saved = lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")
    assert len(saved) == 10
    ids = [json.loads(l)["id"] for l in saved]
    expected = score(load_video_answers(), {i: REPLIES["mock-always-no"] for i in ids})
    assert no["risk_accuracy"] == expected["risk_accuracy"] and no["n"] == 10
    # failed calls are reported, never saved
    assert broken["errors"] == 10 and broken["n"] == 0
    assert lines(tmp_path / "t" / "mock-error" / "responses.jsonl") == []
    # rerun: nothing new is called or written; growing the limit only adds the new videos
    run(CFG, limit=10, results_dir=tmp_path)
    assert len(lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")) == 10
    run(CFG, limit=15, results_dir=tmp_path)
    assert len(lines(tmp_path / "t" / "mock-always-no" / "responses.jsonl")) == 15


def test_limit_order_is_mixed_and_fixed(tmp_path):
    from asimov.data.video import load_video_items
    from asimov.runner import video_order
    first = [i.id for i in video_order(load_video_items())[:20]]
    assert first == [i.id for i in video_order(load_video_items())[:20]]
    answers = load_video_answers()
    assert {answers[i].risk for i in first} == {"Yes", "No"}
