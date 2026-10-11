"""Gemini via the google-genai SDK (2.28). Reads GEMINI_API_KEY.

Thinking "min": Gemini 2.5 takes a token budget (Pro's minimum is 128; Flash and Flash-Lite accept
0). Gemini 3+ takes a level instead: "minimal" for Flash and Flash-Lite, "low" for Pro, whose
lowest level it is. "default" sends no thinking config (dynamic thinking).
"""
from __future__ import annotations

from google import genai
from google.genai import types

from asimov.bench.parts import ImagePart, Part, TextPart
from asimov.models.base import ModelResponse, Thinking, Timer

MAX_OUTPUT_TOKENS = 8192  # includes thinking tokens


def to_contents(parts: list[Part]) -> list:
    out = []
    for p in parts:
        if isinstance(p, TextPart):
            out.append(types.Part.from_text(text=p.text))
        else:
            out.append(types.Part.from_bytes(data=p.data, mime_type=p.media_type))
    return out


def min_thinking(model: str) -> types.ThinkingConfig:
    pro = "-pro" in model
    if model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=128 if pro else 0)
    return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW if pro else types.ThinkingLevel.MINIMAL)


class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str):
        self.model = model
        self.client = genai.Client()

    def generate(self, parts: list[Part], thinking: Thinking = "min") -> ModelResponse:
        # AFC is the SDK's tool-calling loop; we send no tools, and disabling it silences its warning
        cfg = {"max_output_tokens": MAX_OUTPUT_TOKENS,
               "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True)}
        if thinking == "min":
            cfg["thinking_config"] = min_thinking(self.model)
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
