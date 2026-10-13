"""Shared by every decision client (Part 2): check the request, and turn per-option probabilities
into the same reply an LLM gives, so text_score scores both the same way.

Clients get probabilities keyed by the option values they sent (e.g. "severe"); `decision_reply`
maps them back to the answer-key letters. The reply text is the JSON of each question's argmax
letter under its `<question>_response` key; the probabilities, by letter, go in
`ModelResponse.probs` (saved with the reply, never scored here).
A question the model didn't answer (e.g. a per-question refusal) is left out, so it's scored as
unanswered = wrong.
"""

from __future__ import annotations

import json

from asimov.bench.parts import DecisionPart, Part
from asimov.bench.text_score import response_key

Probs = dict[str, dict[str, float] | None]  # question id -> {option: probability}, None if unanswered


def decision_part(parts: list[Part]) -> DecisionPart:
    """The request's single DecisionPart; anything else is a config error (an LLM prompt sent to a decision model)."""
    if len(parts) != 1 or not isinstance(parts[0], DecisionPart):
        raise TypeError("decision clients take one DecisionPart; use a decision prompt (e.g. decision_full) in the config")
    return parts[0]


def argmax(dist: dict[str, float]) -> str:
    """Highest-probability option; ties go to the first in the order given."""
    return max(dist, key=lambda k: dist[k])


def reply_text(probs: Probs) -> str:
    return json.dumps({response_key(q): argmax(dist) for q, dist in probs.items() if dist})


def to_letters(part: DecisionPart, by_value: Probs) -> Probs:
    """{question: {value: p}} -> {question: {letter: p}}, in the order the options were sent."""
    out: Probs = {}
    for q in part.questions:
        dist = by_value.get(q.id)
        if dist is None:
            out[q.id] = None
            continue
        values = [v for v, _ in q.options]
        if set(dist) != set(values):
            raise ValueError(f"{q.id}: answer options {sorted(dist)} don't match the options sent {values}")
        out[q.id] = {letter: dist[v] for v, letter in zip(values, q.letters, strict=True)}
    return out


def decision_reply(part: DecisionPart, by_value: Probs) -> tuple[str, Probs]:
    """Client probabilities by option value -> (reply text, probabilities by letter)."""
    probs = to_letters(part, by_value)
    return reply_text(probs), probs
