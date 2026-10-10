"""Gemini via the google-genai SDK (2.28). Reads GEMINI_API_KEY.

Thinking: Gemini 2.5 Pro can't turn thinking off; its minimum budget is 128 tokens. Flash and
Flash-Lite accept 0. "default" sends no thinking config (dynamic thinking).
"""
from __future__ import annotations

from google import genai
from google.genai import types

from asimov.bench.video import ImagePart, Part, TextPart
from asimov.models.base import ModelResponse, Thinking, Timer

MAX_OUTPUT_TOKENS = 8192  # includes thinking tokens
MIN_THINKING_BUDGET = {"gemini-2.5-pro": 128}  # everything else: 0


def to_contents(parts: list[Part]) -> list:
    out = []
    for p in parts:
        if isinstance(p, TextPart):
            out.append(types.Part.from_text(text=p.text))
        else:
            out.append(types.Part.from_bytes(data=p.data, mime_type=p.media_type))
    return out


def min_budget(model: str) -> int:
    return next((b for prefix, b in MIN_THINKING_BUDGET.items() if model.startswith(prefix)), 0)


class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str):
        self.model = model
        self.client = genai.Client()

    def generate(self, parts: list[Part], thinking: Thinking = "min") -> ModelResponse:
        cfg = {"max_output_tokens": MAX_OUTPUT_TOKENS}
        if thinking == "min":
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=min_budget(self.model))
        with Timer() as t:
            r = self.client.models.generate_content(
                model=self.model, contents=to_contents(parts), config=types.GenerateContentConfig(**cfg)
            )
        u = r.usage_metadata
        return ModelResponse(
            text=r.text or "",
            model=r.model_version or self.model,
            usage={"input_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count,
                   "thinking_tokens": u.thoughts_token_count,
                   "finish_reason": str(r.candidates[0].finish_reason) if r.candidates else None},
            latency_s=t.seconds,
            raw=r.model_dump(mode="json", exclude_none=True),
        )


def list_model_ids() -> list[str]:
    return [m.name.removeprefix("models/") for m in genai.Client().models.list()]
