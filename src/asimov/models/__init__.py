"""Model clients. `get_client("openai", "gpt-5")` etc. SDKs are imported lazily so a missing
key or package for one provider never blocks the others."""
from __future__ import annotations

# The models in the paper's Fig. 5b, as API model IDs. Availability is checked by
# scripts/check_models.py; substitutions are recorded in WORKLOG.md, never made silently.
PAPER_MODELS = {
    # Gemini 2.5 models return 404 for new API keys (checked 2026-10-10). Stand-ins chosen by
    # Tyler: the nearest successors served to this key.
    "Gemini 2.5 Pro": ("gemini", "gemini-3.1-pro-preview"),
    "Gemini 2.5 Flash": ("gemini", "gemini-3-flash-preview"),
    "Gemini 2.5 Flash-Lite": ("gemini", "gemini-3.1-flash-lite"),
    # Opus 4.1 and Sonnet 4 are retired (checked 2026-10-10). Stand-ins chosen by Tyler: the
    # nearest successors still served. Dated IDs so the exact model is recorded.
    "Claude Opus 4.1": ("anthropic", "claude-opus-4-5-20251101"),
    "Claude Sonnet 4": ("anthropic", "claude-sonnet-4-5-20250929"),
    "GPT-5": ("openai", "gpt-5"),
    "GPT-5 Mini": ("openai", "gpt-5-mini"),
    "GPT-5 Nano": ("openai", "gpt-5-nano"),
}


def client_class(provider: str):
    if provider == "anthropic":
        from asimov.models.anthropic_client import AnthropicClient
        return AnthropicClient
    if provider == "openai":
        from asimov.models.openai_client import OpenAIClient
        return OpenAIClient
    if provider == "gemini":
        from asimov.models.gemini_client import GeminiClient
        return GeminiClient
    if provider == "mock":
        from asimov.models.mock import MockClient
        return MockClient
    if provider == "systemone":
        from asimov.models.systemone import SystemOneClient
        return SystemOneClient
    if provider == "openai_decisions":
        from asimov.models.openai_decisions import OpenAIDecisionsClient
        return OpenAIDecisionsClient
    if provider == "cloudflare":
        from asimov.models.cloudflare_clef import CloudflareClefClient
        return CloudflareClefClient
    if provider == "mock_decision":
        from asimov.models.mock import MockDecisionClient
        return MockDecisionClient
    if provider == "local_hf":
        from asimov.models.local_hf import LocalHFClient
        return LocalHFClient
    if provider == "nli":
        from asimov.models.nli import NLIClient
        return NLIClient
    raise ValueError(f"unknown provider {provider}")


def get_client(provider: str, model: str, **service):
    """service: base_url, key_env, params, for decision services and local models only (Part 2; asimov.config.ModelSpec).
    Passed through only when given, so LLM clients are built as `client_class(provider)(model)`."""
    options = {k: v for k, v in service.items() if v is not None}
    return client_class(provider)(model, **options)


def request_settings(provider: str) -> frozenset[str]:
    """Config settings that change what this provider's client sends (asimov.reuse uses it)."""
    return client_class(provider).request_settings
