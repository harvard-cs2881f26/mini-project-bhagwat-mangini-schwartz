"""Decision-model interface (Part 2): prompt parsing, decision requests, probs on replies.

Decision models answer each question with a probability per option. The text benchmark parses
each verbatim prompt into scenario, instruction and the 4 questions, and renders a DecisionPart
from it; decision clients return the same 4-letter JSON as LLMs, so text_score is unchanged.
"""

import json

import pytest
import yaml

from asimov.bench.parts import DecisionPart, DecisionQuestion
from asimov.bench.text import DECISION_PROMPTS, build_request, option_name, parse_decision_prompt, prompt_version, wording
from asimov.bench.text_score import OPTIONS, QUESTIONS, response_key, score, score_item
from asimov.data.text import load_text_answers, load_text_items, parse_prompt
from asimov.models import get_client
from asimov.models.decision import Decided, decision_reply, generate_decision, reply_text
from asimov.models.reply_info import finish
from asimov.runner import ROOT, run


def test_every_prompt_parses_into_scenario_instruction_and_four_questions():
    for item in load_text_items():
        p = parse_prompt(item.prompt)
        assert p.scenario and p.instruction, item.id
        assert tuple(q.key for q in p.questions) == QUESTIONS
        assert [len(q.options) for q in p.questions] == [5, 4, 4, 4], item.id
        for q in p.questions:
            assert "".join(q.options) == OPTIONS[q.key], item.id
            assert all(text.strip() for text in q.options.values()), item.id
            assert q.text.strip(), item.id


@pytest.mark.full_data
def test_all_319_prompts_parse():
    assert len([parse_prompt(i.prompt) for i in load_text_items()]) == 319


def test_parsed_parts_come_from_the_prompt():
    item = load_text_items()[0]
    p = parse_prompt(item.prompt)
    assert p.scenario in item.prompt and p.instruction in item.prompt
    for q in p.questions:
        assert all(text in item.prompt for text in q.options.values())
        assert not any(line.lstrip().startswith(("A. ", "B. ")) for line in q.text.splitlines())
    q4_end = "Choose the letter for the corresponding severity level.\nInjury Severity Rubric (for Q3)"
    assert p.questions[3].text.endswith(q4_end)
    assert p.questions[1].options["D"].startswith("Severe Injury:")
    assert p.questions[2].options["C"].startswith("Possibility of injury is the same")


def test_unparseable_prompt_fails_loudly():
    with pytest.raises(ValueError, match="SCENARIO"):
        parse_prompt("no scenario here")


@pytest.mark.parametrize("prompt", DECISION_PROMPTS)
def test_decision_request_has_state_and_four_questions_with_the_papers_options(prompt):
    for item in load_text_items():
        p = parse_prompt(item.prompt)
        [part] = build_request(item, prompt)
        assert isinstance(part, DecisionPart)
        assert part.state == f"SCENARIO: {p.scenario}\nINSTRUCTION: {p.instruction}"  # no task preamble
        assert tuple(q.id for q in part.questions) == QUESTIONS
        for dq, q in zip(part.questions, p.questions, strict=True):
            assert dq.letters == tuple(q.options)
            assert [d for _, d in dq.options] == list(q.options.values())  # the paper's option text
            assert len({v for v, _ in dq.options}) == len(dq.options)  # values unique within the question
            assert dq.instructions.strip()


def test_option_values_are_short_names_or_letters():
    assert option_name("Burns, Chemical") == "burns_chemical"
    item = load_text_items()[0]
    [named] = build_request(item, "decision_short")
    [lettered] = build_request(item, "decision_short_letters")
    values = {q.id: [v for v, _ in q.options] for q in named.questions}
    q1_text = parse_prompt(item.prompt).questions[0].options.values()
    assert values["latent_risk"] == [option_name(text) for text in q1_text]
    assert values["latent_risk"][-1] == "no_realistic_injury_possible"
    assert values["latent_severity"] == values["activated_risk"] == ["no_injury", "minor", "moderate", "severe"]
    assert values["effect"] == ["eliminates", "reduces", "no_change", "increases"]
    assert all([v for v, _ in q.options] == list(q.letters) for q in lettered.questions)


def test_full_and_short_wording_differ_only_in_instructions():
    item = load_text_items()[0]
    [full], [short] = build_request(item, "decision_full"), build_request(item, "decision_short")
    assert full.state == short.state
    assert [q.options for q in full.questions] == [q.options for q in short.questions]
    p = parse_prompt(item.prompt)
    assert [q.instructions for q in full.questions] == [q.text for q in p.questions]
    assert all(len(s.instructions) < len(f.instructions) for s, f in zip(short.questions, full.questions, strict=True))


def test_decision_prompt_versions_differ_and_cover_the_wording_file():
    versions = {prompt_version(p) for p in DECISION_PROMPTS}
    assert len(versions) == len(DECISION_PROMPTS)
    assert all(v.startswith("asimov_v2_injuries/0.1.0|decision_") for v in versions)
    assert all(len(v.rsplit("|", 1)[1]) == 12 for v in versions)  # hash of prompts/decision_questions.yaml


# --- probabilities -> the same reply shape as an LLM ---


def test_reply_text_is_the_argmax_letter_json_and_scores_like_an_llm_reply():
    probs = {
        "latent_risk": {"A": 0.1, "B": 0.0, "C": 0.0, "D": 0.8, "E": 0.1},
        "latent_severity": {"A": 0.2, "B": 0.2, "C": 0.5, "D": 0.1},
        "effect": {"A": 0.0, "B": 0.9, "C": 0.1, "D": 0.0},
        "activated_risk": {"A": 0.6, "B": 0.4, "C": 0.0, "D": 0.0},
    }
    text = reply_text(probs)
    assert json.loads(text) == {
        response_key("latent_risk"): "D",
        response_key("latent_severity"): "C",
        response_key("effect"): "B",
        response_key("activated_risk"): "A",
    }
    gold = {"latent_risk": "D", "latent_severity": "C", "effect": "B", "activated_risk": "A"}
    outcome = score_item(text, gold)
    assert outcome["parse_ok"] and set(outcome["parse"].values()) == {"ok"}
    assert score({"x": gold}, {"x": text})["latent_risk_accuracy"] == 1.0


def test_unanswered_question_is_left_out_of_the_reply():
    text = reply_text({"latent_risk": {"A": 1.0}, "latent_severity": None})
    assert json.loads(text) == {response_key("latent_risk"): "A"}


def test_probabilities_map_back_from_option_values_to_letters():
    [part] = build_request(load_text_items()[0], "decision_short")
    by_value = {
        q.id: {v: (0.9 if i == 2 else 0.1 / (len(q.options) - 1)) for i, (v, _) in enumerate(q.options)} for q in part.questions
    }
    text, probs = decision_reply(part, by_value)
    assert json.loads(text) == {response_key(q): "C" for q in QUESTIONS}
    assert probs["latent_severity"] == {
        "A": pytest.approx(0.1 / 3),
        "B": pytest.approx(0.1 / 3),
        "C": 0.9,
        "D": pytest.approx(0.1 / 3),
    }


def test_answer_with_unknown_options_fails_loudly():
    [part] = build_request(load_text_items()[0], "decision_short")
    with pytest.raises(ValueError, match="don't match the options sent"):
        decision_reply(part, {"effect": {"A": 0.5, "B": 0.5}})


def test_mock_decision_client_returns_probs_and_rejects_text_requests():
    item = load_text_items()[0]
    client = get_client("mock_decision", "mock-decide-first")
    r = client.generate(build_request(item, DECISION_PROMPTS[0]))
    assert set(r.probs) == set(QUESTIONS)
    for q, dist in r.probs.items():
        assert list(dist) == list(OPTIONS[q]) and sum(dist.values()) == pytest.approx(1)
    assert {q: score_item(r.text, dict.fromkeys(QUESTIONS, "A"))[q]["pred"] for q in QUESTIONS} == dict.fromkeys(QUESTIONS, "A")
    with pytest.raises(TypeError, match="DecisionPart"):
        client.generate(build_request(item, "text_verbatim"))


def test_mock_decision_run_end_to_end_saves_probs(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "smoke_mock_decision.yaml").read_text())
    [row] = [r for r in run(cfg, results_dir=tmp_path) if r["model"] == "mock-decide-first"]
    n = len(load_text_answers())
    assert row["n"] == n and row["parse_failure_rate"] == 0
    saved = [json.loads(line) for line in open(tmp_path / cfg["run"] / "mock-decide-first" / "responses.jsonl")]
    assert len(saved) == n and all(set(r["probs"]) == set(QUESTIONS) for r in saved)


# --- question modes: do a decision model's questions see its own earlier answers? ---


def test_decision_prompt_names_parse_into_wording_letters_and_mode():
    assert parse_decision_prompt("decision_short") == ("short", False, "independent")
    assert parse_decision_prompt("decision_full") == ("full", False, "independent")
    assert parse_decision_prompt("decision_short_letters") == ("short", True, "independent")
    assert parse_decision_prompt("decision_full_letters_cumulative") == ("full", True, "cumulative")
    for bad in (
        "decision_medium",
        "decision_short_chained",
        "decision_short_chained_letters",
        "decision_short_independent",
        "text_verbatim",
    ):
        with pytest.raises(ValueError):
            parse_decision_prompt(bad)
    assert {"decision_short", "decision_full", "decision_short_letters"} <= set(DECISION_PROMPTS)
    assert all(parse_decision_prompt(p) for p in DECISION_PROMPTS)
    assert len(DECISION_PROMPTS) == 8  # {short, full} x {names, letters} x {independent, cumulative}


def test_existing_decision_prompt_versions_are_pinned():
    """Main-run versions: editing a mode's context lines (or a YAML comment) must not move them.
    (They moved once, from ...|126e444fcb3b, when the hash went from the file's bytes to the sections
    a request uses; no decision replies had been saved then.)"""
    assert prompt_version("decision_short") == "asimov_v2_injuries/0.1.0|decision_short|f89d5ee7e6bc"
    assert prompt_version("decision_full") == "asimov_v2_injuries/0.1.0|decision_full|f89d5ee7e6bc"
    assert prompt_version("decision_short_letters") == "asimov_v2_injuries/0.1.0|decision_short_letters|f89d5ee7e6bc"
    assert prompt_version("decision_full_letters") == "asimov_v2_injuries/0.1.0|decision_full_letters|f89d5ee7e6bc"


def test_context_wording_is_in_the_cumulative_versions_only(monkeypatch):
    before = {p: prompt_version(p) for p in DECISION_PROMPTS}
    edited = {**wording(), "context": {**wording()["context"], "latent_risk": "EDITED: {answer}"}}
    monkeypatch.setattr("asimov.bench.text.wording", lambda: edited)
    for p in DECISION_PROMPTS:
        moved = prompt_version(p) != before[p]
        assert moved == (parse_decision_prompt(p).mode != "independent"), p


def test_every_decision_prompt_builds_a_request_with_its_mode():
    item = load_text_items()[0]
    for p in DECISION_PROMPTS:
        [part] = build_request(item, p)
        assert part.mode == parse_decision_prompt(p).mode
        assert tuple(q.id for q in part.questions) == QUESTIONS
        assert part.state == build_request(item, "decision_short")[0].state  # context is added per call, not here


class Scripted:
    """A decision client whose pick per question is fixed (option index), or None to refuse."""

    model = "scripted"

    def __init__(self, picks: dict[str, int | None], usage=None, served=("served-1", "served-2", "served-3", "served-4")):
        self.picks, self.parts = picks, []
        self.usage = usage or {"input_tokens": 10, "output_tokens": 1}
        self.served = served

    def decide(self, part: DecisionPart) -> Decided:
        self.parts.append(part)
        by_value = {}
        for q in part.questions:
            top = self.picks.get(q.id, 0)
            values = [v for v, _ in q.options]
            by_value[q.id] = (
                None if top is None else {v: (0.7 if i == top else 0.3 / (len(values) - 1)) for i, v in enumerate(values)}
            )
        n = len(self.parts)
        return Decided(by_value=by_value, raw={"call": n}, usage=dict(self.usage), model=self.served[n - 1])


def option_text(q: DecisionQuestion, index: int) -> str:
    return q.options[index][1]


def context_line(qid: str, answer: str) -> str:
    return wording()["context"][qid].format(answer=answer)


def test_independent_mode_is_one_call_with_the_raw_unchanged():
    [part] = build_request(load_text_items()[0], "decision_short")
    client = Scripted({})
    r = generate_decision(client, [part])
    assert client.parts == [part]
    assert r.raw == {"call": 1} and r.usage == {"input_tokens": 10, "output_tokens": 1} and r.model == "served-1"


def test_cumulative_mode_adds_every_earlier_pick_in_order():
    [part] = build_request(load_text_items()[0], "decision_short_cumulative")
    client = Scripted({"latent_risk": 1, "latent_severity": 2, "effect": 3, "activated_risk": 0})
    r = generate_decision(client, [part])
    assert [[q.id for q in p.questions] for p in client.parts] == [[q] for q in QUESTIONS]
    q1, q2, q3, _ = part.questions
    lines = [
        context_line("latent_risk", option_text(q1, 1)),
        context_line("latent_severity", option_text(q2, 2)),
        context_line("effect", option_text(q3, 3)),
    ]
    for i, sent in enumerate(client.parts):
        assert sent.state == "\n".join([part.state, *lines[:i]])
    assert json.loads(r.text) == {
        response_key("latent_risk"): "B",
        response_key("latent_severity"): "C",
        response_key("effect"): "D",
        response_key("activated_risk"): "A",
    }
    assert list(r.probs) == list(QUESTIONS) and r.probs["latent_risk"]["B"] == 0.7
    assert r.usage == {"input_tokens": 40, "output_tokens": 4}
    assert r.model == "served-1" and r.latency_s >= 0
    assert r.raw == {"mode": "cumulative", "calls": [{"call": n} for n in (1, 2, 3, 4)]}


def test_unknown_token_counts_stay_unknown_when_summed():
    [part] = build_request(load_text_items()[0], "decision_short_cumulative")
    r = generate_decision(Scripted({}, usage={"input_tokens": None, "output_tokens": 0}), [part])
    assert r.usage == {"input_tokens": None, "output_tokens": 0}


def test_cumulative_mode_with_an_unanswered_q1_adds_no_injury_line():
    [part] = build_request(load_text_items()[0], "decision_short_cumulative")
    client = Scripted({"latent_risk": None, "latent_severity": 1})
    r = generate_decision(client, [part])
    q2_line = context_line("latent_severity", option_text(part.questions[1], 1))
    assert client.parts[1].state == part.state  # Q2 call: no injury line
    assert client.parts[2].state == part.state + "\n" + q2_line  # Q3 call: only Q2's pick
    assert all(context_line("latent_risk", "") not in p.state for p in client.parts)
    assert response_key("latent_risk") not in json.loads(r.text) and r.probs["latent_risk"] is None
    calls = [
        {"answers": {"latent_risk": {"type": "refusal"}}},
        {"answers": {q: {"type": "choice", "probabilities": {}} for q in QUESTIONS[1:]}},
    ]
    assert finish("systemone", {"mode": "cumulative", "calls": calls}) == "refused"
    assert finish("systemone", {"mode": "cumulative", "calls": calls[1:]}) == "complete"


def test_context_comes_from_the_models_pick_never_the_answer_key(monkeypatch):
    item = load_text_items()[0]
    key = load_text_answers()[item.id]["latent_risk"]
    pick = next(i for i, letter in enumerate("ABCDE") if letter != key)
    [part] = build_request(item, "decision_short_cumulative")

    def no_answers(*a, **k):
        raise AssertionError("decision requests must not read the answer key")

    monkeypatch.setattr("asimov.bench.text.load_text_answers", no_answers)
    monkeypatch.setattr("asimov.data.text.load_text_answers", no_answers)
    client = Scripted({"latent_risk": pick})
    generate_decision(client, [part])
    assert client.parts[1].state.endswith(context_line("latent_risk", option_text(part.questions[0], pick)))


def test_mock_decision_cumulative_run_end_to_end_saves_probs(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "smoke_mock_decision_cumulative.yaml").read_text())
    assert parse_decision_prompt(cfg["prompt"]).mode == "cumulative"
    [row] = [r for r in run(cfg, results_dir=tmp_path) if r["model"] == "mock-decide-first"]
    assert row["n"] == len(load_text_answers()) and row["parse_failure_rate"] == 0
    saved = [json.loads(line) for line in open(tmp_path / cfg["run"] / "mock-decide-first" / "responses.jsonl")]
    assert all(set(r["probs"]) == set(QUESTIONS) for r in saved)
    assert all(r["raw"]["mode"] == "cumulative" and len(r["raw"]["calls"]) == 4 for r in saved)
