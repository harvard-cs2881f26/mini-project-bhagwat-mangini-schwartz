"""Mock clients: no network, no keys, fixed replies. For testing the pipeline end to end.

They can't be "perfect" models, because a perfect mock would need the answer key, and clients
never see answers. Scorer tests check exact numbers with hand-built replies instead.
"""
from __future__ import annotations

from asimov.bench.parts import Part
from asimov.models.base import ModelResponse, Thinking
from asimov.models.decision import decision_part, decision_reply

REPLIES = {
    "mock-always-no": '{"risk_of_physical_injury": "No", "injury_severity": "No Harm", '
    '"last_intervention_timestamp_seconds": -1}',
    "mock-always-yes": '{"risk_of_physical_injury": "Yes", "injury_severity": "Moderate Harm", '
    '"last_intervention_timestamp_seconds": 4.0}',
    "mock-garbage": "I think the video shows a kitchen.",
    "mock-refuse": "I'm sorry, I can't help with that.",
}


class MockClient:
    provider = "mock"

    def __init__(self, model: str):
        if model not in REPLIES and model != "mock-error":
            raise ValueError(f"unknown mock {model}; choose from {sorted(REPLIES)}")
        self.model = model

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse:
        if self.model == "mock-error":
            raise RuntimeError("simulated API failure")
        return ModelResponse(text=REPLIES[self.model], model=self.model, usage={"input_tokens": 0, "output_tokens": 0})


class MockDecisionClient:
    """Decision-model mock: answers every question from the options alone, with probabilities.

    mock-decide-first:   0.7 on the first option, the rest shared equally
    mock-decide-uniform: equal probabilities (argmax = first option)
    """
    provider = "mock_decision"
    MODELS = ("mock-decide-first", "mock-decide-uniform")

    def __init__(self, model: str):
        if model not in self.MODELS:
            raise ValueError(f"unknown decision mock {model}; choose from {self.MODELS}")
        self.model = model

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse:
        request = decision_part(parts)
        text, probs = decision_reply(request, {q.id: self.dist([v for v, _ in q.options]) for q in request.questions})
        return ModelResponse(text=text, model=self.model, usage={"input_tokens": 0, "output_tokens": 0}, probs=probs)

    def dist(self, values: list[str]) -> dict[str, float]:
        if self.model == "mock-decide-uniform":
            return {v: 1 / len(values) for v in values}
        rest = 0.3 / (len(values) - 1)
        return {v: 0.7 if i == 0 else rest for i, v in enumerate(values)}
