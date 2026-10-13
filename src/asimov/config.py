"""Run configs (configs/*.yaml), checked when a run starts.

An unknown key or a missing field fails before any model is called, with pydantic's message
naming the field, so a typo like `thinkng: default` can't silently fall back to a default.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from asimov.models.base import Thinking

PARAMS_PROVIDERS = ("systemone", "local_hf", "nli")  # providers whose models take `params`


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str  # label in tables and figures, e.g. "Claude Opus 4.1 (4.5)" for a stand-in
    provider: Literal["anthropic", "openai", "gemini", "mock",  # asimov.models.get_client
                      "mock_decision", "systemone", "openai_decisions", "local_hf", "nli"]
    model: str
    # Decision services (Part 2). base_url: the /v1/systemone service (Jev, Codiv, Kev on Modal);
    # key_env: the env var holding its key; params: extra request fields (e.g. {think: 512}) or
    # our own options ({option_order: reversed}). All three are part of the cache key when set.
    # Local models (local_hf, nli) take params only: repo, revision (and mode, template for local_hf).
    base_url: str | None = None
    key_env: str | None = None
    params: dict[str, StrictInt | StrictStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def systemone_needs_a_service(self) -> ModelSpec:
        extras = [k for k in ("base_url", "key_env") if getattr(self, k)]
        extras += ["params"] if self.params and self.provider not in PARAMS_PROVIDERS else []
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

    @model_validator(mode="after")
    def local_decision_models_need_a_local_prompt(self) -> RunConfig:
        """Local option-probability and NLI models render the request with the wording file's `local`
        section, which only `_local` prompts hash into their version (asimov.bench.text); local
        generation takes a text prompt."""
        from asimov.bench.text import is_local_prompt

        for m in self.models:
            mode = m.params.get("mode") if m.provider == "local_hf" else "logprob" if m.provider == "nli" else None
            if mode == "logprob" and not is_local_prompt(self.prompt):
                raise ValueError(f"{m.name}: {m.provider} option probabilities need a `_local` decision prompt "
                                 f"(e.g. decision_short_local), not {self.prompt!r}")
            if mode == "generate" and is_local_prompt(self.prompt):
                raise ValueError(f"{m.name}: local_hf generate takes a text prompt (e.g. text_verbatim), not {self.prompt!r}")
        return self
