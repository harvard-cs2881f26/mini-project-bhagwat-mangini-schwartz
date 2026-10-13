"""Provider-neutral request parts. Every benchmark's `build_request` returns a list of these,
and each provider client turns them into its own message format.

LLM clients take TextPart / ImagePart. Decision clients (Part 2) take one DecisionPart: a state
plus questions with fixed options, each answered with a probability per option. Its `mode` says
whether the questions see the model's own earlier answers (models.decision.generate_decision)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# independent (non-cumulative): one call, every question answered in isolation (main runs)
# cumulative: one call per question, each with the model's own picks for every earlier question
Mode = Literal["independent", "cumulative"]
MODES: tuple[Mode, ...] = ("independent", "cumulative")


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
    # cumulative: the state line naming the model's own pick for this question, for later
    # questions; "{answer}" = the picked option's description. Never sent with this question itself.
    context: str = ""
    # one-shot prompts only (local base model): a made-up worked example shown before this question
    example: DecisionExample | None = None


@dataclass(frozen=True)
class DecisionExample:
    """A made-up, answered example of a question (never from the dataset): its own state, option
    descriptions in order, and the letter answering it (prompts/decision_questions.yaml `oneshot`)."""

    state: str
    options: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class DecisionPart:
    state: str
    questions: tuple[DecisionQuestion, ...]
    mode: Mode = "independent"


Part = TextPart | ImagePart | DecisionPart
