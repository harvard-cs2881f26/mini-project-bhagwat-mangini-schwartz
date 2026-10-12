"""Run configs (configs/*.yaml), checked when a run starts.

An unknown key or a missing field fails before any model is called, with pydantic's message
naming the field, so a typo like `thinkng: default` can't silently fall back to a default.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from asimov.models.base import Thinking


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str  # label in tables and figures, e.g. "Claude Opus 4.1 (4.5)" for a stand-in
    provider: Literal["anthropic", "openai", "gemini", "mock"]  # asimov.models.get_client
    model: str


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str  # results/<run>/; one run name = one config (asimov.runner.ConfigChanged)
    benchmark: str
    prompt: str
    thinking: Thinking
    models: list[ModelSpec]
