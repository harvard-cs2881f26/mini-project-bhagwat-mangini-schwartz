"""Shared types for model clients.

Every client takes the provider-neutral request parts from `asimov.bench.*.build_request` and
returns the raw reply text plus metadata. Clients never see answer keys. They raise on API
errors; the runner decides what to retry and never caches a failure.

thinking:
    "min"      lowest thinking each model allows (our reproduction default; WORKLOG 2026-10-10)
    "default"  send no thinking/reasoning parameter at all; the provider decides
Temperature is never set: provider defaults (the paper states nothing, and GPT-5 rejects it).

log_reasoning (config flag, default off): ask for the provider's reasoning/thought summary and
return it in `ModelResponse.reasoning`, never in `text`, so scorers never see it. Off sends
exactly what was sent before the flag existed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

from asimov.bench.parts import Part

Thinking = Literal["min", "default"]


@dataclass
class ModelResponse:
    text: str
    model: str  # model ID the provider reports it served
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)  # full provider response, for the shared results
    reasoning: str = ""  # reasoning/thought summary, only with log_reasoning; never scored
    probs: dict = field(default_factory=dict)  # decision models: {question: {option: p}}; empty for LLMs


class ModelClient(Protocol):
    provider: str
    model: str

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse: ...


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
