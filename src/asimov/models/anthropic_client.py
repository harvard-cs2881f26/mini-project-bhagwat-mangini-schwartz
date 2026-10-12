"""Claude via the Anthropic Messages API (anthropic SDK 1.11).

Thinking: the paper's Claude models (Opus 4.1, Sonnet 4) run without thinking when the
`thinking` parameter is omitted, so "min" and "default" both omit it here.
"""
from __future__ import annotations

import base64

import anthropic

from asimov.bench.parts import Part, TextPart
from asimov.models.base import ModelResponse, Thinking, Timer

MAX_TOKENS = 2048  # the answer is a short JSON object


def to_content(parts: list[Part]) -> list[dict]:
    content = []
    for p in parts:
        if isinstance(p, TextPart):
            content.append({"type": "text", "text": p.text})
        else:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": p.media_type,
                        "data": base64.standard_b64encode(p.data).decode(),
                    },
                }
            )
    return content


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.Anthropic()

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse:
        """log_reasoning sends nothing new (no thinking parameter is sent); it keeps any thinking blocks."""
        with Timer() as t:
            r = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": to_content(parts)}],
            )
        text = "".join(b.text for b in r.content if b.type == "text")
        return ModelResponse(
            text=text,
            model=r.model,
            usage={"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
                   "stop_reason": r.stop_reason},
            latency_s=t.seconds,
            raw=r.to_dict(),
            reasoning="\n\n".join(b.thinking for b in r.content if b.type == "thinking") if log_reasoning else "",
        )


def list_model_ids() -> list[str]:
    return [m.id for m in anthropic.Anthropic().models.list()]
