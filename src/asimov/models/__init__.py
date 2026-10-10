"""Model clients. `get_client("openai", "gpt-5")` etc. SDKs are imported lazily so a missing
key or package for one provider never blocks the others."""
from __future__ import annotations

# The models in the paper's Fig. 5b, as API model IDs. Availability is checked by
# scripts/check_models.py; substitutions are recorded in WORKLOG.md, never made silently.
PAPER_MODELS = {
    "Gemini 2.5 Pro": ("gemini", "gemini-2.5-pro"),
    "Gemini 2.5 Flash": ("gemini", "gemini-2.5-flash"),
    "Gemini 2.5 Flash-Lite": ("gemini", "gemini-2.5-flash-lite"),
    "Claude Opus 4.1": ("anthropic", "claude-opus-4-1"),
    "Claude Sonnet 4": ("anthropic", "claude-sonnet-4-0"),
    "GPT-5": ("openai", "gpt-5"),
    "GPT-5 Mini": ("openai", "gpt-5-mini"),
    "GPT-5 Nano": ("openai", "gpt-5-nano"),
}


def get_client(provider: str, model: str):
    if provider == "anthropic":
        from asimov.models.anthropic_client import AnthropicClient
        return AnthropicClient(model)
    if provider == "openai":
        from asimov.models.openai_client import OpenAIClient
        return OpenAIClient(model)
    if provider == "gemini":
        from asimov.models.gemini_client import GeminiClient
        return GeminiClient(model)
    if provider == "mock":
        from asimov.models.mock import MockClient
        return MockClient(model)
    raise ValueError(f"unknown provider {provider}")
