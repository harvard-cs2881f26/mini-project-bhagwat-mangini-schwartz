"""Decision classifiers on the video benchmark, offline: the request built from the LLM prompt, each
client's body (Jev/systemone, Cloudflare Clef, OpenAI Decisions) over mocked transports, and the reply
scored by the unchanged video scorer. No network, no keys.
"""

import io
import json

import httpx
import pytest
from PIL import Image

from asimov.bench.parts import DecisionPart
from asimov.bench.video import DECISION_QUESTIONS, build_request, load_prompt, prompt_version
from asimov.bench.video_score import parse_reply
from asimov.data.video import load_video_answers, load_video_items
from asimov.models.cloudflare_clef import CloudflareClefClient
from asimov.models.decision import generate_decision
from asimov.models.mock import MockDecisionClient
from asimov.models.openai_decisions import decision_input
from asimov.models.systemone import SystemOneClient


@pytest.fixture
def item():
    return load_video_items()[0]


@pytest.fixture
def part(item) -> DecisionPart:
    [p] = build_request(item, "decision_video_v2")
    return p


def test_decision_request_uses_the_llm_prompt_verbatim(part, item):
    text = load_prompt("video_v2")
    assert part.state in text and part.lead in text
    for q in part.questions:
        assert q.instructions in text  # each Qn. paragraph, word for word
    assert "Respond with only this JSON" not in part.state + part.lead
    assert [f.data for f in part.frames] == [p.read_bytes() for p in item.frame_paths]
    assert [q.reply_key for q in part.questions] == [k for k, _ in DECISION_QUESTIONS]
    assert len(part.questions[2].options) == 17  # -1 and the 16 frame times


def test_decision_request_carries_no_answer(part, item):
    raw = json.loads(next(l for l in open(item.frame_paths[0].parents[2] / "items.jsonl") if item.id in l))["answer_raw"]
    assert raw not in part.state + part.lead
    assert item.id not in part.state + part.lead


def test_decision_prompt_has_its_own_saved_reply_key():
    assert prompt_version("decision_video_v2") != prompt_version("video_v2")
    assert prompt_version("decision_video_v2").endswith(load_prompt("video_v2"))


def test_mock_decision_reply_scores_with_the_video_scorer(part):
    reply = MockDecisionClient("mock-decide-first").generate([part])
    pred = parse_reply(reply.text)
    assert pred is not None and pred.risk == "Yes" and pred.severity == "No Harm" and pred.time_s == -1


def answer(body_questions: dict, top: int = 0) -> dict:
    answers = {}
    for qid, q in body_questions.items():
        values = list(q["criteria"])
        probs = {v: (0.7 if i == top else 0.3 / (len(values) - 1)) for i, v in enumerate(values)}
        answers[qid] = {"type": "choice", "choice": values[top], "probabilities": probs}
    return {"model": "x", "answers": answers, "usage": {"input_tokens": 7000, "output_tokens": 0}}


def test_systemone_sends_frames_as_inline_assets_shrunk_with_the_images_header(part, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"], seen["headers"] = json.loads(request.content), request.headers
        return httpx.Response(200, json=answer(seen["body"]["questions"], top=1))

    monkeypatch.setenv("K", "sk-test")
    client = SystemOneClient("jev-latest", base_url="https://api.example.test", key_env="K",
                             params={"image_width": 498, "image_height": 280}, transport=httpx.MockTransport(handler))
    reply = client.generate([part])
    body = seen["body"]
    assert seen["headers"]["TypeSafe-Danger-Zone-Images"] == "true"
    assert "image_width" not in body and len(body["assets"]) == 16
    for ref in body["assets"]:
        assert body["state"].count(ref) == 1
    assert body["state"].index("t = 0.0 s: [[frame00]]") < body["state"].index("t = 7.5 s: [[frame15]]")
    assert body["state"].endswith(part.lead)
    import base64
    img = Image.open(io.BytesIO(base64.b64decode(body["assets"]["[[frame00]]"]["data_base64"])))
    assert img.size == (498, 280)
    pred = parse_reply(reply.text)
    assert pred.risk == "No" and pred.severity == "Mild Harm" and pred.time_s == 0.0


def test_cloudflare_sends_four_tiles_and_unwraps_the_result(part, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"success": True, "result": answer(seen["body"]["questions"], top=0)})

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    client = CloudflareClefClient("clef", transport=httpx.MockTransport(handler))
    reply = client.generate([part])
    assert seen["url"].endswith("/accounts/acct/ai/run/@cf/cloudflare/clef")
    assert len(seen["body"]["images"]) == 4 and seen["body"]["images"][0].startswith("data:image/jpeg;base64,")
    assert "image 4 holds 6.0 s to 7.5 s" in seen["body"]["state"]
    assert reply.usage["input_tokens"] == 7000
    assert parse_reply(reply.text).risk == "Yes"


def test_openai_decisions_input_interleaves_labels_and_frames(part):
    [msg] = decision_input(part)
    kinds = [c["type"] for c in msg["content"]]
    assert kinds[:3] == ["input_text", "input_text", "input_image"] and kinds[-1] == "input_text"
    assert kinds.count("input_image") == 16
    assert msg["content"][1]["text"] == "t = 0.0 s" and msg["content"][2]["detail"] == "high"


def test_text_decision_requests_are_unchanged():
    from asimov.bench.text import build_request as text_request
    from asimov.data.text import load_text_items
    [p] = text_request(load_text_items()[0], "decision_short")
    assert p.frames == () and decision_input(p) == p.state
    reply = generate_decision(MockDecisionClient("mock-decide-first"), [p])
    assert all(k.endswith("_response") for k in json.loads(reply.text))


def test_every_video_answer_is_an_option(part):
    """Answer-key values the scorer compares against are among the options sent (severity, risk)."""
    options = {q.id: {v for v, _ in q.options} for q in part.questions}
    for a in load_video_answers().values():
        assert a.risk in options["risk_of_physical_injury"]
