"""Cloudflare's Clef decision models on Workers AI (`clef`, 27B; `clef-flash`, 9B): video only.

Same question shape as TypeSafe's /v1/systemone (`state` + `choice` questions with `criteria`), at
POST https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/cloudflare/<model>
with `Authorization: Bearer {CLOUDFLARE_API_TOKEN}`; the answer comes wrapped in {"result": ...}.
Price: $0.24 per M input tokens, outputs free (Workers AI model page, 2026-10-18; clef-flash unlisted).

Images: at most 4 per request, so the 16 frames go as 4 tiles of 2x2 frames with each frame's label
drawn on it (models/frames.py), and the state says how they are laid out. Images are data URLs (the
shape the service accepted on 2026-10-18; the docs don't give one). Each frame reaches the model at
640x360, a quarter of the original area: a known handicap against the full-size LLM requests.
Retries as the systemone client does.
"""

from __future__ import annotations

import base64
import os
import time

import httpx

from asimov.bench.parts import DecisionPart, Part
from asimov.models.base import ModelResponse, Thinking
from asimov.models.decision import Decided, generate_decision, require_answers
from asimov.models.frames import tiles, tile_note
from asimov.models.systemone import ATTEMPTS, BACKOFF_S, RETRY, TIMEOUT_S, choice_probs, retry_after

MODELS = ("clef", "clef-flash")
URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"


class CloudflareClefClient:
    provider = "cloudflare"
    request_settings: frozenset[str] = frozenset()  # no thinking setting

    def __init__(self, model: str, transport: httpx.BaseTransport | None = None):
        if model not in MODELS:
            raise ValueError(f"unknown Clef model {model}; choose from {MODELS}")
        missing = [k for k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN") if k not in os.environ]
        if missing:
            raise KeyError(f"{', '.join(missing)} not set; add to .env")
        self.model, self.params = model, {}
        self.url = URL.format(account=os.environ["CLOUDFLARE_ACCOUNT_ID"], model=model)
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"}, timeout=TIMEOUT_S, transport=transport
        )
        self.sleep = time.sleep

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse:
        return generate_decision(self, parts)

    def decide(self, part: DecisionPart) -> Decided:
        envelope = self.post(request_body(part, self.model))
        raw = envelope.get("result") or {}
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


def request_body(part: DecisionPart, model: str) -> dict:
    questions = {q.id: {"type": "choice", "instructions": q.instructions, "criteria": dict(q.options)} for q in part.questions}
    body: dict = {"model": model, "state": part.state, "questions": questions}
    if part.frames:
        body["state"] = f"{part.state} {tile_note(part.frames)}" + (f"\n\n{part.lead}" if part.lead else "")
        body["images"] = ["data:image/jpeg;base64," + base64.b64encode(t).decode() for t in tiles(part.frames)]
    return body
