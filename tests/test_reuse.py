"""Reusing saved replies across runs when the request a model gets is identical (asimov/reuse.py;
CLI: scripts/reuse_replies.py). Claude's client sends the same request for thinking min/default
and log_reasoning on/off, so its replies can be shared; GPT/Gemini's can't. No network."""

import json

import pytest

from asimov.models import request_settings
from asimov.reuse import copy_rows, differing_settings, reusable, reuse
from asimov.runner import run

MIN = {
    "run": "a",
    "benchmark": "video",
    "prompt": "video_v2",
    "thinking": "min",
    "log_reasoning": True,
    "models": [{"name": "Always No", "provider": "mock", "model": "mock-always-no"}],
}
DEFAULT = {**MIN, "run": "b", "thinking": "default"}


def test_each_provider_declares_what_changes_its_request():
    assert request_settings("anthropic") == frozenset()
    assert request_settings("openai") == request_settings("gemini") == {"thinking", "log_reasoning"}


def test_differing_settings():
    assert differing_settings(MIN, DEFAULT) == {"thinking"}
    assert differing_settings(MIN, {**MIN, "run": "c"}) == set()


def test_reusable_only_when_the_request_is_identical():
    assert reusable("anthropic", MIN, DEFAULT)
    assert not reusable("openai", MIN, DEFAULT)
    assert not reusable("gemini", MIN, {**MIN, "log_reasoning": False})
    assert reusable("openai", MIN, {**MIN, "run": "c"})  # same settings, only the run name differs


def test_different_prompt_or_benchmark_is_never_reusable():
    assert not reusable("anthropic", MIN, {**DEFAULT, "prompt": "video_v1"})
    assert not reusable("anthropic", MIN, {**DEFAULT, "benchmark": "text"})


def test_copy_rows_rekeys_and_marks_the_source():
    rows = [
        {"id": "x", "key": "old", "text": "t", "usage": {"input_tokens": 1}},
        {"id": "y", "key": "other", "text": "u", "usage": {}},
    ]
    out = copy_rows(rows, src_key="old", dst_key="new", src_run="a", already=set())
    assert out == [
        {"id": "x", "key": "new", "text": "t", "usage": {"input_tokens": 1}, "reused_from": {"run": "a", "key": "old"}}
    ]
    assert copy_rows(rows, "old", "new", "a", already={"x"}) == []


def test_reused_replies_are_scored_and_counted_without_calls(tmp_path):
    run(MIN, limit=3, results_dir=tmp_path)
    copied = reuse(MIN, DEFAULT, results_dir=tmp_path)
    assert copied == {"mock-always-no": 3}
    row = run(DEFAULT, limit=3, results_dir=tmp_path, call_models=False)[0]
    assert row["n"] == 3 and row["n_reused"] == 3
    saved = [json.loads(line) for line in (tmp_path / "b" / "mock-always-no" / "responses.jsonl").read_text().splitlines()]
    assert all(r["reused_from"]["run"] == "a" for r in saved)
    assert reuse(MIN, DEFAULT, results_dir=tmp_path) == {"mock-always-no": 0}  # idempotent


def test_reuse_refuses_models_whose_request_differs(tmp_path):
    cfg_a = {**MIN, "models": [{"name": "GPT", "provider": "openai", "model": "gpt-5"}]}
    cfg_b = {**cfg_a, "run": "b", "thinking": "default"}
    assert reuse(cfg_a, cfg_b, results_dir=tmp_path) == {}  # not asked for by name: skipped
    with pytest.raises(ValueError, match="request differs"):
        reuse(cfg_a, cfg_b, only=["gpt-5"], results_dir=tmp_path)  # asked for by name: refused
