"""Shared by every decision client (Part 2): check the request, and turn per-option probabilities
into the same reply an LLM gives, so text_score scores both the same way.

Clients get probabilities keyed by the option values they sent (e.g. "severe"); `decision_reply`
maps them back to the answer-key letters. The reply text is the JSON of each question's pick
under its `<question>_response` key: the highest-probability option; on an exact tie (luna rounds
probabilities to 2 decimals), the option the service itself returned as its `choice`, else the
first tied option in the paper's order; the probabilities, by letter, go in
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
from dataclasses import dataclass, field
from typing import Protocol

from asimov.bench.parts import DecisionPart, DecisionQuestion, Part
from asimov.bench.text_score import response_key
from asimov.models.base import ModelResponse, Timer

Probs = dict[str, dict[str, float] | None]  # question id -> {option: probability}, None if unanswered
PROB_SUM_TOLERANCE = 0.03  # sum within 1 +- this: Jev and luna round to 2 decimals (5 options: up to 0.025 off; Jev seen at 0.99)


@dataclass
class Decided:
    """One decision call: probabilities by option value, the provider's response, tokens, served model."""

    by_value: Probs
    raw: dict
    usage: dict  # {"input_tokens": int | None, "output_tokens": int | None}
    model: str
    chosen: dict[str, str | None] = field(default_factory=dict)  # the service's own `choice` value per question


def require_answers(part: DecisionPart, answered: set[str]) -> None:
    """A response must hold an answer (choice or refusal) for every question sent; anything else
    (e.g. an error body with status 200) raises, so it's logged as an error and never cached."""
    missing = [q.id for q in part.questions if q.id not in answered]
    if missing:
        raise ValueError(f"response has no answer for {missing}; not saved (rerun to retry)")


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


def reply_text(probs: Probs, picks: dict[str, str | None] | None = None) -> str:
    """The 4-letter reply; `picks` (from `pick`) override the plain argmax where given."""
    picks = picks or {}
    return json.dumps({response_key(q): picks.get(q) or argmax(dist) for q, dist in probs.items() if dist})


def pick(by_letter: dict[str, float] | None, sent_order: list[str], chosen: str | None) -> str | None:
    """The answer letter: the highest probability; on an exact tie the service's own `chosen` letter
    if it is among the tied, else the first tied letter in `sent_order`."""
    if not by_letter:
        return None
    top = max(by_letter.values())
    tied = [letter for letter in sent_order if by_letter[letter] == top]
    return chosen if chosen in tied else tied[0]


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
        if abs(sum(dist.values()) - 1) > PROB_SUM_TOLERANCE:
            raise ValueError(f"{q.id}: probabilities sum to {sum(dist.values()):.4f}, not 1")
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


def context_line(question: DecisionQuestion, letter: str | None) -> str | None:
    """The state line naming the model's own pick (`letter`) for `question`; None if unanswered or no wording."""
    if not letter or not question.context:
        return None
    description = dict(zip(question.letters, (d for _, d in question.options), strict=True))[letter]
    return question.context.format(answer=description)


def picks_for(questions: tuple[DecisionQuestion, ...], step: Probs, chosen: dict[str, str | None]) -> dict[str, str | None]:
    """Each question's answer letter, with ties broken by the service's own `choice` (by value -> letter)."""
    out = {}
    for q in questions:
        to_letter = dict(zip((v for v, _ in q.options), q.letters, strict=True))
        out[q.id] = pick(step.get(q.id), list(q.letters), to_letter.get(chosen.get(q.id) or ""))
    return out


def total(values: list[int | None]) -> int | None:
    """Summed token count; None if any call didn't report it (an unknown total, not a smaller one)."""
    known = [v for v in values if v is not None]
    return sum(known) if len(known) == len(values) else None


def generate_decision(client: DecisionClient, parts: list[Part]) -> ModelResponse:
    """Run a decision request in its mode through `client.decide` and merge the calls into one reply."""
    part = decision_part(parts)
    probs: Probs = {}
    picks: dict[str, str | None] = {}
    calls: list[Decided] = []
    lines: list[str] = []
    with Timer() as t:
        for questions in steps(part):
            sub = part if part.mode == "independent" else DecisionPart("\n".join([part.state, *lines]), questions)
            decided = client.decide(sub)
            step = to_letters(sub, decided.by_value)
            step_picks = picks_for(questions, step, decided.chosen)
            probs.update(step)
            picks.update(step_picks)
            calls.append(decided)
            lines += [line for q in questions if (line := context_line(q, step_picks[q.id]))]
    if part.mode == "independent":
        [only] = calls
        raw, usage = only.raw, only.usage
    else:
        raw = {"mode": part.mode, "calls": [c.raw for c in calls]}
        usage = {k: total([c.usage.get(k) for c in calls]) for k in ("input_tokens", "output_tokens")}
    return ModelResponse(
        text=reply_text(probs, picks), model=calls[0].model, usage=usage, latency_s=t.seconds, raw=raw, probs=probs
    )
