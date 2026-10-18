"""GPT via the OpenAI Responses API (openai SDK 3.24).

Thinking: GPT-5 models always reason; "min" sends the model's lowest effort (`min_effort`:
"minimal" for GPT-5, "none" for gpt-6-luna, whose model page lists no "minimal"), "default" sends
nothing, "high" sends effort "high" and raises the output cap to HIGH_MAX_OUTPUT_TOKENS (reasoning
tokens count against it). Images go as base64 data URLs with detail="high" so they are not downscaled (we send
full resolution; WORKLOG 2026-10-10).
"""
from __future__ import annotations

import base64

import openai

from asimov.bench.parts import Part, TextPart
from asimov.models.base import ModelResponse, Thinking, Timer

MAX_OUTPUT_TOKENS = 8192  # includes reasoning tokens
HIGH_MAX_OUTPUT_TOKENS = 32768
MIN_EFFORT = {"gpt-6-luna": "none"}  # lowest effort per model when it isn't "minimal"


def min_effort(model: str) -> str:
    return MIN_EFFORT.get(model, "minimal")


def to_input(parts: list[Part], system: str | None = None) -> list[dict]:
    content = []
    for p in parts:
        if isinstance(p, TextPart):
            content.append({"type": "input_text", "text": p.text})
        else:
            b64 = base64.standard_b64encode(p.data).decode()
            content.append({"type": "input_image", "image_url": f"data:{p.media_type};base64,{b64}", "detail": "high"})
    messages = [{"role": "user", "content": content}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    return messages


class OpenAIClient:
    provider = "openai"
    request_settings = frozenset({"thinking", "log_reasoning"})  # both change the `reasoning` param sent

    def __init__(self, model: str):
        self.model = model
        self.client = openai.OpenAI()

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False,
                 system: str | None = None, reasoning_effort: str | None = None) -> ModelResponse:
        if reasoning_effort is not None and reasoning_effort not in ("minimal", "low", "medium", "high"):
            raise ValueError(f"Unsupported GPT-5 reasoning effort: {reasoning_effort}")
        reasoning = ({"effort": reasoning_effort} if reasoning_effort is not None
                     else ({"effort": min_effort(self.model)} if thinking == "min" else ({"effort": "high"} if thinking == "high" else {})))
        reasoning |= {"summary": "auto"} if log_reasoning else {}
        kwargs = {"reasoning": reasoning} if reasoning else {}
        with Timer() as t:
            r = self.client.responses.create(
                model=self.model, input=to_input(parts, system), max_output_tokens=HIGH_MAX_OUTPUT_TOKENS if thinking == "high" else MAX_OUTPUT_TOKENS, **kwargs
            )
        u = r.usage
        return ModelResponse(
            text=r.output_text,
            model=r.model,
            usage={"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                   "reasoning_tokens": u.output_tokens_details.reasoning_tokens, "status": r.status},
            latency_s=t.seconds,
            raw=r.to_dict(),
            reasoning=reasoning_summary(r.output) if log_reasoning else "",
        )


def reasoning_summary(output: list) -> str:
    """Text of the reasoning items' summaries (reasoning={"summary": "auto"})."""
    return "\n\n".join(s.text for item in output if item.type == "reasoning" for s in item.summary or [])


def list_model_ids() -> list[str]:
    return [m.id for m in openai.OpenAI().models.list()]
