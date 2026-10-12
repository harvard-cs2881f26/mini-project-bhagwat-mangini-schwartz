"""How a reply ended (complete / truncated / refused) and cached input tokens, read from the
saved provider response (`raw`). Shapes follow real saved responses in results/repro_fig4."""

import pytest

from asimov.models.reply_info import cached_input_tokens, finish


def openai(status="completed", reason=None, content_type="output_text", cached=0):
    return {
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": [{"type": "reasoning", "content": []}, {"type": "message", "content": [{"type": content_type, "text": "x"}]}],
        "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": cached}},
    }


def anthropic(stop_reason="end_turn", cached=0):
    return {"stop_reason": stop_reason, "usage": {"input_tokens": 10, "cache_read_input_tokens": cached}}


def gemini(finish_reason: str | None = "STOP", block_reason: str | None = None, cached: int | None = None):
    raw = {
        "candidates": [{"finish_reason": finish_reason}] if finish_reason else [],
        "usage_metadata": {"prompt_token_count": 10},
    }
    if block_reason:
        raw["prompt_feedback"] = {"block_reason": block_reason}
    if cached is not None:
        raw["usage_metadata"]["cached_content_token_count"] = cached
    return raw


@pytest.mark.parametrize(
    "provider,raw,kind",
    [
        ("openai", openai(), "complete"),
        ("openai", openai("incomplete", "max_output_tokens"), "truncated"),
        ("openai", openai("incomplete", "content_filter"), "refused"),
        ("openai", openai(content_type="refusal"), "refused"),
        ("anthropic", anthropic(), "complete"),
        ("anthropic", anthropic("max_tokens"), "truncated"),
        ("anthropic", anthropic("refusal"), "refused"),
        ("gemini", gemini(), "complete"),
        ("gemini", gemini("MAX_TOKENS"), "truncated"),
        ("gemini", gemini("SAFETY"), "refused"),
        ("gemini", gemini("PROHIBITED_CONTENT"), "refused"),
        ("gemini", gemini(None, block_reason="SAFETY"), "refused"),
        ("mock", {}, "complete"),
    ],
)
def test_finish(provider, raw, kind):
    assert finish(provider, raw) == kind


@pytest.mark.parametrize(
    "provider,raw,n",
    [
        ("openai", openai(cached=1024), 1024),
        ("anthropic", anthropic(cached=512), 512),
        ("gemini", gemini(cached=256), 256),
        ("gemini", gemini(), 0),  # field left out of the saved response when the provider reports none
        ("mock", {}, None),
    ],
)
def test_cached_input_tokens(provider, raw, n):
    assert cached_input_tokens(provider, raw) == n
