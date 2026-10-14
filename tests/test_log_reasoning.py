"""log_reasoning config flag: reasoning/thought summaries requested only when on, saved in their
own `reasoning` field (never in `text`, never scored), and part of the cache key only when on.
Clients are tested against fake SDK objects: no network, no keys."""

import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from asimov.bench import get_benchmark
from asimov.bench.parts import Part, TextPart
from asimov.models import anthropic_client, gemini_client, openai_client
from asimov.models.base import ModelResponse
from asimov.runner import ROOT, run, run_key

PARTS: list[Part] = [TextPart("question")]


class Recorder:
    """Stands in for an SDK method: records the keyword arguments, returns a canned response."""

    def __init__(self, response):
        self.response, self.kwargs = response, None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.response


# --- OpenAI ---


def openai_response(summary: str | None):
    reasoning = NS(type="reasoning", summary=[NS(type="summary_text", text=summary)] if summary else [])
    usage = NS(input_tokens=10, output_tokens=5, output_tokens_details=NS(reasoning_tokens=3))
    return NS(
        output_text='{"a": 1}',
        model="gpt-5-x",
        usage=usage,
        status="completed",
        output=[reasoning, NS(type="message")],
        to_dict=lambda: {},
    )


@pytest.fixture
def fake_openai(monkeypatch):
    create = Recorder(openai_response("Thought about the floor."))
    monkeypatch.setattr(openai_client.openai, "OpenAI", lambda: NS(responses=NS(create=create)))
    return create


def test_openai_summary_only_when_on(fake_openai):
    client = openai_client.OpenAIClient("gpt-5")
    r = client.generate(PARTS, thinking="min")
    assert fake_openai.kwargs["reasoning"] == {"effort": "minimal"} and r.reasoning == ""
    r = client.generate(PARTS, thinking="min", log_reasoning=True)
    assert fake_openai.kwargs["reasoning"] == {"effort": "minimal", "summary": "auto"}
    assert r.reasoning == "Thought about the floor." and r.text == '{"a": 1}'


def test_openai_default_thinking_with_summary(fake_openai):
    openai_client.OpenAIClient("gpt-5").generate(PARTS, thinking="default", log_reasoning=True)
    assert fake_openai.kwargs["reasoning"] == {"summary": "auto"}
    openai_client.OpenAIClient("gpt-5").generate(PARTS, thinking="default")
    assert "reasoning" not in fake_openai.kwargs


# --- Gemini ---


def gemini_response():
    parts = [NS(text="Considered the stairs.", thought=True), NS(text='{"a": 1}', thought=None)]
    return NS(
        text='{"a": 1}',
        model_version="gemini-x",
        usage_metadata=NS(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=3),
        candidates=[NS(finish_reason="STOP", content=NS(parts=parts))],
        model_dump=lambda **_: {},
    )


@pytest.fixture
def fake_gemini(monkeypatch):
    generate = Recorder(gemini_response())
    monkeypatch.setattr(gemini_client.genai, "Client", lambda: NS(models=NS(generate_content=generate)))
    return generate


def test_gemini_thoughts_only_when_on(fake_gemini):
    client = gemini_client.GeminiClient("gemini-3.1-pro-preview")
    r = client.generate(PARTS, thinking="min")
    assert not fake_gemini.kwargs["config"].thinking_config.include_thoughts and r.reasoning == ""
    r = client.generate(PARTS, thinking="min", log_reasoning=True)
    cfg = fake_gemini.kwargs["config"].thinking_config
    assert cfg.include_thoughts is True and cfg.thinking_level == "LOW"  # min level kept
    assert r.reasoning == "Considered the stairs." and r.text == '{"a": 1}'


def test_gemini_default_thinking_with_thoughts(fake_gemini):
    gemini_client.GeminiClient("gemini-3-flash-preview").generate(PARTS, thinking="default", log_reasoning=True)
    assert fake_gemini.kwargs["config"].thinking_config.include_thoughts is True
    gemini_client.GeminiClient("gemini-3-flash-preview").generate(PARTS, thinking="default")
    assert fake_gemini.kwargs["config"].thinking_config is None


# --- Claude ---


@pytest.fixture
def fake_anthropic(monkeypatch):
    blocks = [NS(type="thinking", thinking="Weighed the options."), NS(type="text", text='{"a": 1}')]
    response = NS(
        content=blocks, model="claude-x", usage=NS(input_tokens=10, output_tokens=5), stop_reason="end_turn", to_dict=lambda: {}
    )
    create = Recorder(response)
    monkeypatch.setattr(anthropic_client.anthropic, "Anthropic", lambda: NS(messages=NS(create=create)))
    return create


def test_claude_keeps_thinking_blocks_only_when_on(fake_anthropic):
    client = anthropic_client.AnthropicClient("claude-opus-4-5")
    assert client.generate(PARTS, thinking="min").reasoning == ""
    r = client.generate(PARTS, thinking="min", log_reasoning=True)
    assert "thinking" not in fake_anthropic.kwargs  # nothing new sent: no thinking at min
    assert r.reasoning == "Weighed the options." and r.text == '{"a": 1}'


# --- cache key ---


def test_cache_key_unchanged_when_off_and_different_when_on():
    old = hashlib.sha256(b"gpt-5|pv|min").hexdigest()[:12]  # the formula before this flag existed
    assert run_key("gpt-5", "pv", "min") == old == run_key("gpt-5", "pv", "min", log_reasoning=False)
    assert run_key("gpt-5", "pv", "min", log_reasoning=True) != old


def test_saved_video_replies_keep_their_key():
    summary = json.loads((ROOT / "results/vid0_repro_fig5/gpt-5/summary.json").read_text())
    prompt_version = get_benchmark("video").prompt_version(summary["prompt"])
    assert run_key("gpt-5", prompt_version, summary["thinking"]) == summary["key"]


# --- runner: separate field, not scored, counted ---

CFG = {
    "run": "t",
    "benchmark": "video",
    "prompt": "video_v2",
    "thinking": "min",
    "log_reasoning": True,
    "models": [{"name": "Fake", "provider": "mock", "model": "mock-always-no"}],
}


class ReasoningClient:
    """Reply text is unreadable; the reasoning holds a valid answer that must never be scored."""

    def __init__(self):
        self.flags = []

    def generate(self, parts, thinking="min", log_reasoning=False):
        self.flags.append(log_reasoning)
        answer = '{"risk_of_physical_injury": "Yes", "last_intervention_timestamp_seconds": 2.0}'
        return ModelResponse(text="no answer here", model="m", reasoning=answer if log_reasoning else "")


def test_runner_saves_reasoning_apart_and_never_scores_it(tmp_path, monkeypatch):
    fake = ReasoningClient()
    monkeypatch.setattr("asimov.runner.get_client", lambda provider, model: fake)
    row = run(CFG, limit=2, results_dir=tmp_path)[0]
    assert fake.flags == [True, True]
    saved = [json.loads(line) for line in (tmp_path / "t" / "mock-always-no" / "responses.jsonl").read_text().splitlines()]
    assert all(r["reasoning"].startswith("{") and r["text"] == "no answer here" for r in saved)
    assert row["parse_failure_rate"] == 1  # scored from text only
    assert row["n_with_reasoning"] == 2


def test_log_reasoning_must_be_a_bool(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        run({**CFG, "log_reasoning": "maybe"}, limit=1, results_dir=tmp_path)
