"""OpenAI Decisions API (Part 2: `luna_decide`, gpt-6-luna; docs/apis/openai_decisions.md).

One call per item: `input` = the state, and the 4 questions as `choice` questions named by question
id, with `choices` = [{value, description}] (option value -> the paper's option text). Answers
are matched by `name`, not position, and mapped back to letters (models.decision). A per-question
`refusal` leaves that question unanswered (scored wrong; `finish` = refused). The SDK retries
429/5xx itself (2 retries by default). Decisions has no thinking setting, so `thinking` doesn't apply.
"""

from __future__ import annotations

import openai

from asimov.bench.parts import Part
from asimov.models.base import ModelResponse, Thinking, Timer
from asimov.models.decision import decision_part, decision_reply


class OpenAIDecisionsClient:
    provider = "openai_decisions"

    def __init__(self, model: str, params: dict | None = None):
        if params:
            raise ValueError(f"openai_decisions takes no params, got {params}")
        self.model, self.params = model, {}
        self.client = openai.OpenAI()

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse:
        part = decision_part(parts)
        questions = [
            {
                "type": "choice",
                "name": q.id,
                "instructions": q.instructions,
                "choices": [{"value": v, "description": d} for v, d in q.options],
            }
            for q in part.questions
        ]
        with Timer() as t:
            raw = self.client.decisions.create(model=self.model, input=part.state, questions=questions).to_dict()
        answers = {a.get("name"): a for a in raw.get("answers") or []}
        by_value = {q.id: choice_probs(answers.get(q.id)) for q in part.questions}
        text, probs = decision_reply(part, by_value)
        usage = raw.get("usage") or {}
        return ModelResponse(
            text=text,
            model=raw.get("model", self.model),
            latency_s=t.seconds,
            raw=raw,
            probs=probs,
            usage={"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")},
        )


def choice_probs(answer: dict | None) -> dict[str, float] | None:
    """[{value, probability}] -> {value: probability}; None for a refusal or a missing answer."""
    if not answer or answer.get("type") != "choice":
        return None
    return {p["value"]: p["probability"] for p in answer.get("probabilities") or []}
