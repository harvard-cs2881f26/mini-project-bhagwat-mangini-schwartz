"""Explicit candidate thinking must reach providers and invalidate image caches."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from asimov.bench.image import request_key
from asimov.data.image import ImageInput
from asimov.models.anthropic_client import AnthropicClient
from asimov.models.openai_client import OpenAIClient


def test_openai_effort_override_and_default(monkeypatch):
    calls = []
    response = SimpleNamespace(
        output_text="[]",
        model="gpt-5-mini",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2, output_tokens_details=SimpleNamespace(reasoning_tokens=1)),
        status="completed",
        to_dict=lambda: {},
    )
    client = object.__new__(OpenAIClient)
    client.model = "gpt-5-mini"
    monkeypatch.setattr(
        client,
        "client",
        SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or response)),
        raising=False,
    )
    client.generate([], thinking="default")
    client.generate([], thinking="default", reasoning_effort="high")
    assert "reasoning" not in calls[0]
    assert calls[1]["reasoning"] == {"effort": "high"}
    with pytest.raises(ValueError):
        client.generate([], reasoning_effort="typo")


@pytest.mark.parametrize("budget", [1024, 4096])
def test_claude_budget_keeps_answer_headroom(budget, monkeypatch):
    calls = []
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="[]")],
        model="claude-sonnet-4",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
        stop_reason="end_turn",
        to_dict=lambda: {},
    )
    client = object.__new__(AnthropicClient)
    client.model = "claude-sonnet-4"
    monkeypatch.setattr(
        client,
        "client",
        SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or response)),
        raising=False,
    )
    assert client.generate([], thinking_budget=budget, system="Rules").text == "[]"
    assert calls[0]["thinking"] == {"type": "enabled", "budget_tokens": budget}
    assert calls[0]["max_tokens"] == budget + 2048
    assert calls[0]["system"] == "Rules"
    for invalid in (True, 1023, "4096"):
        with pytest.raises(ValueError):
            client.generate([], thinking_budget=cast(Any, invalid))


def test_explicit_settings_have_separate_keys_and_legacy_keys_are_stable(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    item = ImageInput("image_0000", "Point <1>", image, 10, 10)
    args = (item, "openai", "gpt-5-mini", "default", "image", "system_user")
    legacy = request_key(*args)
    assert request_key(*args, {}) == legacy
    assert request_key(*args, {"reasoning_effort": "high"}) != legacy
    assert request_key(*args, {"reasoning_effort": "high"}) != request_key(*args, {"reasoning_effort": "medium"})
