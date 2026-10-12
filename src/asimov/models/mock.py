"""Mock clients: no network, no keys, fixed replies. For testing the pipeline end to end.

They can't be "perfect" models, because a perfect mock would need the answer key, and clients
never see answers. Scorer tests check exact numbers with hand-built replies instead.
"""
from __future__ import annotations

from asimov.bench.parts import Part
from asimov.models.base import ModelResponse, Thinking

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
