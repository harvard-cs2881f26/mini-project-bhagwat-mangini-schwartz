"""Shared by every decision client (Part 2): check the request, and turn per-option probabilities
into the same reply an LLM gives, so text_score scores both the same way.

Clients get probabilities keyed by the option values they sent (e.g. "severe"); `decision_reply`
maps them back to the answer-key letters. The reply text is the JSON of each question's argmax
letter under its `<question>_response` key; the probabilities, by letter, go in
`ModelResponse.probs` (saved with the reply, never scored here).
A question the model didn't answer (e.g. a per-question refusal) is left out, so it's scored as
unanswered = wrong.

Question modes live here, in one place (`generate_decision`): each client only makes one call
(`decide`) for the questions it is given. The request's `mode` sets the calls:
    independent  (non-cumulative) one call with all 4 questions (raw = that call's response, as before)
    cumulative   one call per question; each state adds a line per earlier question's own pick
A pick is the argmax option's description (the paper's option text), put into that question's
`context` line from prompts/decision_questions.yaml. It comes from the model's reply, never the
answer key; an unanswered question adds no line. Cumulative raw = {"mode", "calls": [...]}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from asimov.bench.parts import DecisionPart, DecisionQuestion, Part
from asimov.bench.text_score import response_key
from asimov.models.base import ModelResponse, Timer

Probs = dict[str, dict[str, float] | None]  # question id -> {option: probability}, None if unanswered


@dataclass
class Decided:
    """One decision call: probabilities by option value, the provider's response, tokens, served model."""

    by_value: Probs
    raw: dict
    usage: dict  # {"input_tokens": int | None, "output_tokens": int | None}
    model: str


class DecisionClient(Protocol):
    model: str

    def decide(self, part: DecisionPart) -> Decided: ...


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


def steps(part: DecisionPart) -> list[tuple[DecisionQuestion, ...]]:
    """The questions asked in each call, in order, for the request's mode."""
    qs = part.questions
    if part.mode == "cumulative":
        return [(q,) for q in qs]
    return [qs]


def context_line(question: DecisionQuestion, by_letter: dict[str, float] | None) -> str | None:
    """The state line naming the model's own pick for `question`; None if unanswered or no wording."""
    if not by_letter or not question.context:
        return None
    description = dict(zip(question.letters, (d for _, d in question.options), strict=True))[argmax(by_letter)]
    return question.context.format(answer=description)


def total(values: list[int | None]) -> int | None:
    """Summed token count; None if any call didn't report it (an unknown total, not a smaller one)."""
    known = [v for v in values if v is not None]
    return sum(known) if len(known) == len(values) else None


def generate_decision(client: DecisionClient, parts: list[Part]) -> ModelResponse:
    """Run a decision request in its mode through `client.decide` and merge the calls into one reply."""
    part = decision_part(parts)
    probs: Probs = {}
    calls: list[Decided] = []
    lines: list[str] = []
    with Timer() as t:
        for questions in steps(part):
            sub = part if part.mode == "independent" else DecisionPart("\n".join([part.state, *lines]), questions)
            decided = client.decide(sub)
            step = to_letters(sub, decided.by_value)
            probs.update(step)
            calls.append(decided)
            lines += [line for q in questions if (line := context_line(q, step[q.id]))]
    if part.mode == "independent":
        [only] = calls
        raw, usage = only.raw, only.usage
    else:
        raw = {"mode": part.mode, "calls": [c.raw for c in calls]}
        usage = {k: total([c.usage.get(k) for c in calls]) for k in ("input_tokens", "output_tokens")}
    return ModelResponse(text=reply_text(probs), model=calls[0].model, usage=usage, latency_s=t.seconds, raw=raw, probs=probs)
