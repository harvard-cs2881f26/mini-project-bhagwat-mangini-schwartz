"""Run the local model clients on a Modal GPU (params `backend: modal`); no modal or torch import here.

The same client code runs on both sides. On Modal (models/modal_app.py), a container builds the
ordinary client with `backend: local` and serves it through `serve_decide` / `serve_complete`; on
this machine the client sends each request through `remote_decide` / `remote_complete` instead of
loading a model. Requests and answers cross as plain dicts (DecisionPart <-> dict).

Saved rows record where the reply was made: raw["remote"] = {backend, device, gpu, compute_s,
load_s}; the row's latency_s is the round trip. The container renders questions from its own copy
of the wording file, so each answer carries a hash of its `local` section and a reply made with
different wording than this checkout's raises (redeploy the app) instead of being saved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from asimov.bench.parts import DecisionExample, DecisionPart, DecisionQuestion
from asimov.models.base import ModelResponse, Timer
from asimov.models.decision import Decided

APP = "asimov-local-models"  # modal deploy -m asimov.models.modal_app (docs in that module)
BACKENDS = ("local", "modal")


def local_wording_hash() -> str:
    from asimov.models.local_text import local_wording

    return hashlib.sha256(json.dumps(local_wording(), sort_keys=True).encode()).hexdigest()[:12]


def part_to_dict(part: DecisionPart) -> dict:
    return asdict(part)


def part_from_dict(d: dict) -> DecisionPart:
    questions = tuple(
        DecisionQuestion(
            id=q["id"],
            instructions=q["instructions"],
            options=tuple((v, desc) for v, desc in q["options"]),
            letters=tuple(q["letters"]),
            context=q["context"],
            example=example_from_dict(q.get("example")),
        )
        for q in d["questions"]
    )
    return DecisionPart(state=d["state"], questions=questions, mode=d["mode"])


def example_from_dict(d: dict | None) -> DecisionExample | None:
    return DecisionExample(state=d["state"], options=tuple(d["options"]), answer=d["answer"]) if d else None


# --- server side (inside the Modal container) ---


def serve_decide(client: Any, part: dict, info: dict) -> dict:
    """Run the client's own `decide` on a request dict; `info` = where it ran (device, gpu, load_s)."""
    with Timer() as t:
        d = client.decide(part_from_dict(part))
    return {**asdict(d), "remote": {**info, "compute_s": t.seconds}, "local_wording": local_wording_hash()}


def serve_complete(client: Any, text: str, info: dict) -> dict:
    r = client.complete(text)
    return {**asdict(r), "remote": {**info, "compute_s": r.latency_s}}


# --- client side (this machine) ---


def check_wording(answer: dict) -> None:
    if answer["local_wording"] != local_wording_hash():
        raise RuntimeError(
            "the Modal app renders questions with different `local` wording than this checkout; redeploy it: "
            "PYTHONPATH=src modal deploy -m asimov.models.modal_app"
        )


def remote_decide(remote: Any, part: DecisionPart) -> Decided:
    answer = remote.decide.remote(part_to_dict(part))
    check_wording(answer)
    raw = {**answer["raw"], "remote": answer["remote"]}
    return Decided(answer["by_value"], raw, answer["usage"], answer["model"], answer["chosen"])


def remote_complete(remote: Any, text: str) -> ModelResponse:
    with Timer() as t:
        answer = remote.complete.remote(text)
    raw = {**answer["raw"], "remote": answer["remote"]}
    return ModelResponse(text=answer["text"], model=answer["model"], usage=answer["usage"], latency_s=t.seconds, raw=raw)


def remote_model(cls_name: str, **params: str) -> Any:
    """The deployed Modal class for these params (one container per model; modal_app.py)."""
    import modal

    return modal.Cls.from_name(APP, cls_name)(**params)
