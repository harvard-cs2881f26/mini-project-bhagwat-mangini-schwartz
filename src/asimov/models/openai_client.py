"""GPT via the OpenAI Responses API (openai SDK 3.24).

Thinking: GPT-5 models always reason; "min" sends reasoning.effort="minimal", "default" sends
nothing. Images go as base64 data URLs with detail="high" so they are not downscaled (we send
full resolution; WORKLOG 2026-10-10).
"""
from __future__ import annotations

import base64

import openai

from asimov.bench.parts import ImagePart, Part, TextPart
from asimov.models.base import ModelResponse, Thinking, Timer

MAX_OUTPUT_TOKENS = 8192  # includes reasoning tokens


def to_input(parts: list[Part]) -> list[dict]:
    content = []
    for p in parts:
        if isinstance(p, TextPart):
            content.append({"type": "input_text", "text": p.text})
        else:
            b64 = base64.standard_b64encode(p.data).decode()
            content.append({"type": "input_image", "image_url": f"data:{p.media_type};base64,{b64}", "detail": "high"})
    return [{"role": "user", "content": content}]


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str):
        self.model = model
        self.client = openai.OpenAI()

    def generate(self, parts: list[Part], thinking: Thinking = "min") -> ModelResponse:
        kwargs = {"reasoning": {"effort": "minimal"}} if thinking == "min" else {}
        with Timer() as t:
            r = self.client.responses.create(
                model=self.model, input=to_input(parts), max_output_tokens=MAX_OUTPUT_TOKENS, **kwargs
            )
        u = r.usage
        return ModelResponse(
            text=r.output_text,
            model=r.model,
            usage={"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                   "reasoning_tokens": u.output_tokens_details.reasoning_tokens, "status": r.status},
            latency_s=t.seconds,
            raw=r.to_dict(),
        )


def list_model_ids() -> list[str]:
    return [m.id for m in openai.OpenAI().models.list()]
