"""Run configs (configs/*.yaml), checked when a run starts.

An unknown key or a missing field fails before any model is called, with pydantic's message
naming the field, so a typo like `thinkng: default` can't silently fall back to a default.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from asimov.models.base import Thinking


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str  # label in tables and figures, e.g. "Claude Opus 4.1 (4.5)" for a stand-in
    provider: Literal["anthropic", "openai", "gemini", "mock",  # asimov.models.get_client
                      "mock_decision", "systemone", "openai_decisions"]
    model: str
    # Decision services (Part 2). base_url: the /v1/systemone service (Jev, Codiv, Kev on Modal);
    # key_env: the env var holding its key; params: extra request fields (e.g. {think: 512}) or
    # our own options ({option_order: reversed}). All three are part of the cache key when set.
    base_url: str | None = None
    key_env: str | None = None
    params: dict[str, StrictInt | StrictStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def systemone_needs_a_service(self) -> ModelSpec:
        extras = [k for k in ("base_url", "key_env", "params") if getattr(self, k)]
        if self.provider != "systemone" and extras:
            raise ValueError(f"{extras} only apply to provider systemone, not {self.provider}")
        if self.provider == "systemone" and not (self.base_url and self.key_env):
            raise ValueError("a systemone model needs base_url (e.g. https://api.typesafe.ai) and key_env "
                             "(e.g. TYPESAFE_API_KEY)")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str  # results/<run>/; one run name = one config (asimov.runner.ConfigChanged)
    benchmark: str
    prompt: str
    thinking: Thinking
    log_reasoning: StrictBool = False  # ask for reasoning/thought summaries (asimov.models.base); part of the cache key
    models: list[ModelSpec]
