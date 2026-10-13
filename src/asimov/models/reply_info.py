"""Facts about a reply that only the provider's response says, read from the saved `raw`.

Pure functions over the full response each client saves (`ModelResponse.raw`), so they work on
replies saved before this module existed (`scripts/run.py --no-calls`) and need no new request
parameters. The runner records them per item and per model for every benchmark.

finish(provider, raw):
    "truncated"  cut off by the output-token limit (OpenAI status "incomplete" for
                 max_output_tokens, Claude stop_reason "max_tokens", Gemini finish_reason MAX_TOKENS)
    "refused"    a refusal or safety block (OpenAI refusal content or content_filter, Claude
                 stop_reason "refusal", Gemini prompt_feedback.block_reason or a safety finish_reason;
                 decision models: any question answered with something other than a choice, in
                 any call of a chained/cumulative reply)
    "complete"   anything else
"""

from __future__ import annotations

from typing import Literal

Finish = Literal["complete", "truncated", "refused"]
FINISHES: tuple[Finish, ...] = ("complete", "truncated", "refused")

GEMINI_REFUSED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


def _openai(raw: dict) -> Finish:
    reason = (raw.get("incomplete_details") or {}).get("reason")
    if raw.get("status") == "incomplete":
        return "refused" if reason == "content_filter" else "truncated"
    contents = [c for item in raw.get("output") or [] if item.get("type") == "message" for c in item.get("content") or []]
    return "refused" if any(c.get("type") == "refusal" for c in contents) else "complete"


ANTHROPIC_STOPS: dict[str, Finish] = {"max_tokens": "truncated", "refusal": "refused"}


def _anthropic(raw: dict) -> Finish:
    return ANTHROPIC_STOPS.get(raw.get("stop_reason") or "", "complete")


def _gemini(raw: dict) -> Finish:
    if (raw.get("prompt_feedback") or {}).get("block_reason"):
        return "refused"
    reasons = {str(c.get("finish_reason")) for c in raw.get("candidates") or []}
    if reasons & GEMINI_REFUSED:
        return "refused"
    return "truncated" if "MAX_TOKENS" in reasons else "complete"


def _decision(raw: dict) -> Finish:
    """Decision services: answers keyed by question (systemone) or a list with names (OpenAI).
    Chained/cumulative replies ({"mode", "calls"}): refused if any call was."""
    if "calls" in raw:
        return "refused" if any(_decision(c) == "refused" for c in raw["calls"]) else "complete"
    answers = raw.get("answers") or {}
    answers = list(answers.values()) if isinstance(answers, dict) else answers
    return "refused" if any(a.get("type") != "choice" for a in answers) else "complete"


def finish(provider: str, raw: dict) -> Finish:
    reader = {"openai": _openai, "anthropic": _anthropic, "gemini": _gemini,
              "systemone": _decision, "openai_decisions": _decision}.get(provider)
    return reader(raw) if reader else "complete"


def cached_input_tokens(provider: str, raw: dict) -> int | None:
    """Input tokens served from the provider's prompt cache; None if the provider doesn't say."""
    if provider == "openai_decisions" and "calls" in raw:  # chained/cumulative: summed over calls
        return sum(cached_input_tokens(provider, c) or 0 for c in raw["calls"])
    if provider in ("openai", "openai_decisions"):
        return ((raw.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens") or 0
    if provider == "anthropic":
        return (raw.get("usage") or {}).get("cache_read_input_tokens") or 0
    if provider == "gemini":  # the saved response leaves the field out when there was no cache hit
        return (raw.get("usage_metadata") or {}).get("cached_content_token_count") or 0
    return None
