"""Decision models over TypeSafe's `/v1/systemone` protocol (Part 2): Jev (TypeSafe), OpenJev and
Laya (Codiv), Kev-4B (self-deployed on Modal). One client; each service is a config row with its
own base_url, key_env and params (docs/apis/systemone.md).

One call per item: `state` + the 4 questions as `choice` questions keyed by question id, each with
`criteria` = option value -> the paper's option text. Answers are mapped back by value
(models.decision), so the reply scores like an LLM's. Questions are answered in isolation by the
service; cumulative prompts make one call per question per item, run by models.decision.generate_decision
(`decide` is one call). Retries 408/429/5xx/529 with exponential backoff, honouring `retry-after`; other errors
raise with the service's message (a 422 names the bad field).

params: request fields passed through (e.g. OpenJev `think`: 512), except our own options:
    option_order: "paper" (default) | "reversed"  -- options sent in reverse, to measure order bias
`thinking` doesn't apply: these services have no thinking setting (OpenJev's is `think` in params).
"""

from __future__ import annotations

import os
import time

import httpx

from asimov.bench.parts import DecisionPart, Part
from asimov.models.base import ModelResponse, Thinking
from asimov.models.decision import Decided, generate_decision, require_answers

PATH = "/v1/systemone"
RETRY = {408, 429, 500, 502, 503, 504, 529}
ATTEMPTS = 5  # first try + 4 retries
BACKOFF_S = (1, 2, 4, 8)
TIMEOUT_S = 120  # a Modal cold start alone takes ~35 s (Kev README)
OPTION_ORDERS = ("paper", "reversed")
OUR_PARAMS = ("option_order",)


class SystemOneClient:
    provider = "systemone"
    request_settings: frozenset[str] = frozenset()  # thinking and log_reasoning are ignored: decision services have neither

    def __init__(
        self,
        model: str,
        base_url: str | None,
        key_env: str | None,
        params: dict | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        if not base_url or not key_env:
            raise ValueError("SystemOneClient needs base_url and key_env (set them in the config)")
        if key_env not in os.environ:
            raise KeyError(f"{key_env} is not set; add it to .env (docs/apis/systemone.md §2)")
        self.model, self.params = model, params or {}
        if self.params.get("option_order", "paper") not in OPTION_ORDERS:
            raise ValueError(f"option_order must be one of {OPTION_ORDERS}")
        self.url = base_url.rstrip("/") + PATH
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {os.environ[key_env]}"}, timeout=TIMEOUT_S, transport=transport
        )
        self.sleep = time.sleep

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse:
        return generate_decision(self, parts)

    def decide(self, part: DecisionPart) -> Decided:
        raw = self.post(request_body(part, self.model, self.params))
        answers = raw.get("answers") or {}
        require_answers(part, set(answers))
        usage = raw.get("usage") or {}
        return Decided(
            by_value={q.id: choice_probs(answers.get(q.id)) for q in part.questions},
            raw=raw,
            usage={"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")},
            model=raw.get("model", self.model),
            chosen={q.id: (answers.get(q.id) or {}).get("choice") for q in part.questions},
        )

    def post(self, body: dict) -> dict:
        for attempt in range(ATTEMPTS):
            r = self.http.post(self.url, json=body)
            if r.status_code not in RETRY or attempt == ATTEMPTS - 1:
                break
            self.sleep(retry_after(r) or BACKOFF_S[attempt])
        if r.is_error:
            raise httpx.HTTPStatusError(f"{r.status_code} from {self.url}: {r.text[:500]}", request=r.request, response=r)
        return r.json()


def request_body(part: DecisionPart, model: str, params: dict) -> dict:
    reverse = params.get("option_order") == "reversed"
    questions = {
        q.id: {"type": "choice", "instructions": q.instructions, "criteria": dict(reversed(q.options) if reverse else q.options)}
        for q in part.questions
    }
    passthrough = {k: v for k, v in params.items() if k not in OUR_PARAMS}
    return {"state": part.state, "model": model, "questions": questions, **passthrough}


def choice_probs(answer: dict | None) -> dict[str, float] | None:
    """A choice answer's probabilities by option value; None for anything else (refusal, missing)."""
    if not answer or answer.get("type") != "choice":
        return None
    return answer.get("probabilities")


def retry_after(r: httpx.Response) -> float | None:
    try:
        return float(r.headers["retry-after"])
    except (KeyError, ValueError):
        return None
