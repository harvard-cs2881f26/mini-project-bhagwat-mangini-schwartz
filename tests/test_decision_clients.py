"""Hosted decision clients (Part 2), offline: /v1/systemone over a mocked HTTP transport, OpenAI
Decisions through a fake SDK client. Response shapes follow the providers' docs
(docs/apis/systemone.md §6, docs/apis/openai_decisions.md §3); no network, no keys.
"""

import json

import httpx
import pytest
import yaml

from asimov.bench.text import build_request
from asimov.bench.text_score import OPTIONS, QUESTIONS, response_key
from asimov.config import ModelSpec
from asimov.data.text import load_text_items
from asimov.models.openai_decisions import OpenAIDecisionsClient
from asimov.models.reply_info import cached_input_tokens, finish
from asimov.models.systemone import SystemOneClient
from asimov.runner import ROOT, run, run_key

KEY = "SYSTEMONE_TEST_KEY"


@pytest.fixture
def part():
    [p] = build_request(load_text_items()[0], "decision_short")
    return p


def peaked(values: list[str], top: int) -> dict[str, float]:
    """0.7 on values[top], the rest shared: a valid distribution with a known argmax."""
    rest = 0.3 / (len(values) - 1)
    return {v: 0.7 if i == top else rest for i, v in enumerate(values)}


def systemone_answer(body: dict, top: int = 1) -> dict:
    """A /v1/systemone response for `body`, shaped like the TypeSafe docs example."""
    answers = {}
    for qid, q in body["questions"].items():
        probs = peaked(list(q["criteria"]), top)
        answers[qid] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": 0.6}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 1200, "output_tokens": 0}}


def systemone_client(handler, monkeypatch, params=None) -> SystemOneClient:
    monkeypatch.setenv(KEY, "sk-test")
    client = SystemOneClient(
        "jev-latest",
        base_url="https://api.example.test",
        key_env=KEY,
        params=params or {},
        transport=httpx.MockTransport(handler),
    )
    client.sleep = lambda s: None  # no real waiting in retry tests
    return client


# --- /v1/systemone ---


def test_systemone_request_snapshot(part, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=systemone_answer(json.loads(request.content)))

    systemone_client(handler, monkeypatch, params={"think": 512}).generate([part])
    [req] = seen
    assert req.method == "POST" and str(req.url) == "https://api.example.test/v1/systemone"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == "jev-latest" and body["think"] == 512
    assert body["state"] == part.state
    assert list(body["questions"]) == list(QUESTIONS)
    for q in part.questions:
        sent = body["questions"][q.id]
        assert sent["type"] == "choice" and sent["instructions"] == q.instructions
        assert sent["criteria"] == dict(q.options)  # value -> the paper's option text, in paper order


def test_systemone_reply_is_argmax_letters_with_probs(part, monkeypatch):
    client = systemone_client(lambda r: httpx.Response(200, json=systemone_answer(json.loads(r.content))), monkeypatch)
    r = client.generate([part])
    assert json.loads(r.text) == {response_key(q): "B" for q in QUESTIONS}  # option index 1 = letter B
    for q, dist in r.probs.items():
        assert list(dist) == list(OPTIONS[q]) and sum(dist.values()) == pytest.approx(1)
    assert r.model == "jev-1.13.0" and r.usage == {"input_tokens": 1200, "output_tokens": 0}
    assert finish("systemone", r.raw) == "complete"


def test_reversed_option_order_maps_back_by_value(part, monkeypatch):
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json=systemone_answer(body, top=0))  # always the first option shown

    r = systemone_client(handler, monkeypatch, params={"option_order": "reversed"}).generate([part])
    [body] = sent
    assert "option_order" not in body  # our parameter, not the API's
    assert list(body["questions"]["effect"]["criteria"]) == ["increases", "no_change", "reduces", "eliminates"]
    assert json.loads(r.text)[response_key("effect")] == "D"  # first shown = "increases" = letter D
    assert json.loads(r.text)[response_key("latent_risk")] == "E"


@pytest.mark.parametrize("status", [429, 529, 503])
def test_systemone_retries_rate_limits_and_overload(part, monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(status, headers={"retry-after": "0"}, json={"error": "busy"})
        return httpx.Response(200, json=systemone_answer(json.loads(request.content)))

    systemone_client(handler, monkeypatch).generate([part])
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 401, 422])
def test_systemone_client_errors_are_not_retried_and_say_why(part, monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json={"error": {"message": "questions.effect: bad criteria"}})

    with pytest.raises(httpx.HTTPStatusError, match="bad criteria"):
        systemone_client(handler, monkeypatch).generate([part])
    assert len(calls) == 1


def test_systemone_gives_up_after_repeated_overload(part, monkeypatch):
    client = systemone_client(lambda r: httpx.Response(529, json={"error": "overloaded"}), monkeypatch)
    with pytest.raises(httpx.HTTPStatusError, match="529"):
        client.generate([part])


def test_systemone_unanswered_question_counts_as_unanswered(part, monkeypatch):
    def handler(request):
        out = systemone_answer(json.loads(request.content))
        out["answers"]["effect"] = {"type": "refusal"}
        return httpx.Response(200, json=out)

    r = systemone_client(handler, monkeypatch).generate([part])
    assert response_key("effect") not in json.loads(r.text) and r.probs["effect"] is None
    assert finish("systemone", r.raw) == "refused"


def test_systemone_needs_its_key(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    with pytest.raises(KeyError, match=KEY):
        SystemOneClient("jev-latest", base_url="https://api.example.test", key_env=KEY)


def test_systemone_rejects_llm_prompts(part, monkeypatch):
    client = systemone_client(lambda r: httpx.Response(500), monkeypatch)
    with pytest.raises(TypeError, match="DecisionPart"):
        client.generate(build_request(load_text_items()[0], "text_verbatim"))


# --- OpenAI Decisions ---


class FakeDecision:
    def __init__(self, data: dict):
        self.data = data

    def to_dict(self) -> dict:
        return self.data


class FakeDecisions:
    def __init__(self, refuse: str | None = None):
        self.calls, self.refuse = [], refuse

    def create(self, **kwargs):
        self.calls.append(kwargs)
        answers = []
        for q in reversed(kwargs["questions"]):  # answer order isn't promised: match by name
            if q["name"] == self.refuse:
                answers.append({"type": "refusal", "name": q["name"]})
                continue
            probs = peaked([c["value"] for c in q["choices"]], 2)
            answers.append(
                {
                    "type": "choice",
                    "name": q["name"],
                    "choice": max(probs, key=probs.get),
                    "confidence": 0.5,
                    "probabilities": [{"value": v, "probability": p} for v, p in probs.items()],
                }
            )
        return FakeDecision(
            {
                "model": "gpt-6-luna-2026-10-01",
                "answers": answers,
                "usage": {"input_tokens": 900, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0}},
            }
        )


def decisions_client(fake: FakeDecisions) -> OpenAIDecisionsClient:
    client = OpenAIDecisionsClient.__new__(OpenAIDecisionsClient)
    client.model, client.params = "gpt-6-luna", {}
    client.client = type("SDK", (), {"decisions": fake})()
    return client


def test_decisions_request_snapshot_uses_the_same_wording(part):
    fake = FakeDecisions()
    decisions_client(fake).generate([part])
    [call] = fake.calls
    assert call["model"] == "gpt-6-luna" and call["input"] == part.state
    for sent, q in zip(call["questions"], part.questions, strict=True):
        assert sent == {
            "type": "choice",
            "name": q.id,
            "instructions": q.instructions,
            "choices": [{"value": v, "description": d} for v, d in q.options],
        }


def test_decisions_reply_matches_answers_by_name(part):
    r = decisions_client(FakeDecisions()).generate([part])
    assert json.loads(r.text) == {response_key(q): "C" for q in QUESTIONS}
    assert all(sum(d.values()) == pytest.approx(1) for d in r.probs.values())
    assert r.model == "gpt-6-luna-2026-10-01" and finish("openai_decisions", r.raw) == "complete"


def test_decisions_per_question_refusal_is_unanswered(part):
    r = decisions_client(FakeDecisions(refuse="latent_severity")).generate([part])
    assert response_key("latent_severity") not in json.loads(r.text)
    assert finish("openai_decisions", r.raw) == "refused"


# --- config and cache key ---


def test_systemone_spec_needs_base_url_and_key_env():
    with pytest.raises(ValueError, match="base_url"):
        ModelSpec(name="Jev", provider="systemone", model="jev-latest")


def test_cache_key_covers_service_and_params_and_is_unchanged_without_them():
    assert run_key("gpt-5", "v", "min") == run_key("gpt-5", "v", "min", extra="")
    jev = run_key("jev-latest", "v", "min", extra="https://api.typesafe.ai")
    codiv = run_key("jev-latest", "v", "min", extra="https://api.codiv.ai")
    think = run_key("jev-latest", "v", "min", extra='https://api.codiv.ai|{"think": 512}')
    assert len({jev, codiv, think}) == 3


def test_cache_hit_makes_no_call(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=systemone_answer(json.loads(request.content)))

    real = httpx.Client

    monkeypatch.setattr(
        "asimov.models.systemone.httpx.Client", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(handler)})
    )
    monkeypatch.setenv(KEY, "sk-test")
    cfg = yaml.safe_load(f"""
run: test_systemone
benchmark: text
prompt: decision_short
thinking: min
models:
  - {{name: Jev, provider: systemone, model: jev-latest, base_url: "https://api.example.test", key_env: {KEY}}}
""")
    run(cfg, limit=3, results_dir=tmp_path)
    run(cfg, limit=3, results_dir=tmp_path)
    assert len(calls) == 3
    rows = [json.loads(line) for line in open(tmp_path / "txt0_test_systemone" / "jev-latest" / "responses.jsonl")]
    assert len(rows) == 3 and all(set(r["probs"]) == set(QUESTIONS) for r in rows)


def test_hosted_part2_configs_load():
    for path in sorted((ROOT / "configs").glob("part2_*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        for spec in cfg["models"]:
            ModelSpec.model_validate(spec)


# --- cumulative question mode through the hosted clients ---


def earlier_lines(part, top: int, upto: int) -> list[str]:
    """State lines naming each earlier question's pick (option index `top`), for questions before `upto`."""
    return [q.context.format(answer=q.options[top][1]) for q in part.questions[:upto]]


def test_systemone_cumulative_sends_one_question_per_call_with_its_own_picks(monkeypatch):
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json=systemone_answer(body, top=1))

    [part] = build_request(load_text_items()[0], "decision_short_cumulative")
    r = systemone_client(handler, monkeypatch).generate([part])
    assert [list(b["questions"]) for b in sent] == [[q] for q in QUESTIONS]
    for i, body in enumerate(sent):
        assert body["state"] == "\n".join([part.state, *earlier_lines(part, 1, i)])
    assert json.loads(r.text) == {response_key(q): "B" for q in QUESTIONS}
    assert r.usage == {"input_tokens": 4800, "output_tokens": 0} and r.model == "jev-1.13.0"
    assert r.raw["mode"] == "cumulative" and len(r.raw["calls"]) == 4
    assert finish("systemone", r.raw) == "complete" and cached_input_tokens("systemone", r.raw) is None


def test_decisions_cumulative_sums_cached_tokens_and_flags_refusals():
    [part] = build_request(load_text_items()[0], "decision_short_cumulative")
    fake = FakeDecisions(refuse="effect")
    r = decisions_client(fake).generate([part])
    assert [[q["name"] for q in c["questions"]] for c in fake.calls] == [[q] for q in QUESTIONS]
    # the refused Q3 adds no line to Q4's state
    assert fake.calls[3]["input"] == "\n".join([part.state, *earlier_lines(part, 2, 2)])
    assert response_key("effect") not in json.loads(r.text) and json.loads(r.text)[response_key("latent_risk")] == "C"
    assert r.usage == {"input_tokens": 3600, "output_tokens": 0}
    assert finish("openai_decisions", r.raw) == "refused"
    for c in r.raw["calls"]:
        c["usage"]["input_tokens_details"]["cached_tokens"] = 100
    assert cached_input_tokens("openai_decisions", r.raw) == 400


# --- review fixes (2026-10-13) ---


def test_tie_goes_to_the_services_own_choice(part, monkeypatch):
    def handler(request):
        out = systemone_answer(json.loads(request.content))
        values = list(json.loads(request.content)["questions"]["latent_severity"]["criteria"])
        out["answers"]["latent_severity"] = {
            "type": "choice",
            "choice": values[1],
            "probabilities": {values[0]: 0.47, values[1]: 0.47, values[2]: 0.06, values[3]: 0.0},
        }
        return httpx.Response(200, json=out)

    r = systemone_client(handler, monkeypatch).generate([part])
    assert json.loads(r.text)[response_key("latent_severity")] == "B"  # the service chose B, tied with A


def test_200_without_answers_is_an_error_not_a_reply(part, monkeypatch):
    client = systemone_client(lambda r: httpx.Response(200, json={"error": "proxy timeout"}), monkeypatch)
    with pytest.raises(ValueError, match="no answer for"):
        client.generate([part])


def test_probabilities_must_sum_to_one(part, monkeypatch):
    def handler(request):
        out = systemone_answer(json.loads(request.content))
        out["answers"]["effect"]["probabilities"] = dict.fromkeys(out["answers"]["effect"]["probabilities"], 0.5)
        return httpx.Response(200, json=out)

    with pytest.raises(ValueError, match="sum to"):
        systemone_client(handler, monkeypatch).generate([part])


def test_service_fields_only_on_systemone():
    with pytest.raises(ValueError, match="only apply to provider systemone"):
        ModelSpec(name="x", provider="openai", model="gpt-5", params={"think": 512})
