"""Provider-neutral request parts. Every benchmark's `build_request` returns a list of these,
and each provider client turns them into its own message format.

LLM clients take TextPart / ImagePart. Decision clients (Part 2) take one DecisionPart: a state
plus questions with fixed options, each answered with a probability per option."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"


@dataclass(frozen=True)
class DecisionQuestion:
    id: str  # question key, e.g. "latent_severity"; answers are mapped back by it
    instructions: str
    options: tuple[tuple[str, str], ...]  # (value, description) in the order sent; answers map back by value
    letters: tuple[str, ...]  # the answer-key letter of each option, aligned with `options`; never sent


@dataclass(frozen=True)
class DecisionPart:
    state: str
    questions: tuple[DecisionQuestion, ...]


Part = TextPart | ImagePart | DecisionPart
