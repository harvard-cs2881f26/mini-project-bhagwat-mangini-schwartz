"""Decision-model interface (Part 2): prompt parsing, decision requests, probs on replies.

Decision models answer each question with a probability per option. The text benchmark parses
each verbatim prompt into scenario, instruction and the 4 questions, and renders a DecisionPart
from it; decision clients return the same 4-letter JSON as LLMs, so text_score is unchanged.
"""

import json

import pytest
import yaml

from asimov.bench.parts import DecisionPart
from asimov.bench.text import DECISION_PROMPTS, build_request, prompt_version
from asimov.bench.text_score import OPTIONS, QUESTIONS, response_key, score, score_item
from asimov.data.text import load_text_answers, load_text_items, parse_prompt
from asimov.models import get_client
from asimov.models.decision import reply_text
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
    assert p.questions[3].text.endswith("Choose the letter for the corresponding severity level.\nInjury Severity Rubric (for Q3)")
    assert p.questions[1].options["D"].startswith("Severe Injury:")
    assert p.questions[2].options["C"].startswith("Possibility of injury is the same")


def test_unparseable_prompt_fails_loudly():
    with pytest.raises(ValueError, match="SCENARIO"):
        parse_prompt("no scenario here")


@pytest.mark.parametrize("prompt", DECISION_PROMPTS)
def test_decision_request_has_state_and_four_lettered_questions(prompt):
    item = load_text_items()[0]
    p = parse_prompt(item.prompt)
    [part] = build_request(item, prompt)
    assert isinstance(part, DecisionPart)
    assert p.scenario in part.state and p.instruction in part.state
    assert tuple(q.id for q in part.questions) == QUESTIONS
    for dq, q in zip(part.questions, p.questions, strict=True):
        assert dq.options == tuple(q.options.items())  # letters as values, the paper's option text
        assert dq.instructions.strip()


def test_decision_prompt_versions_differ_and_cover_the_wording_file():
    versions = {prompt_version(p) for p in DECISION_PROMPTS}
    assert len(versions) == len(DECISION_PROMPTS)
    assert all(v.startswith("asimov_v2_injuries/0.1.0|decision_") for v in versions)
    assert all(len(v.rsplit("|", 1)[1]) == 12 for v in versions)  # hash of prompts/decision_questions.yaml


# --- probabilities -> the same reply shape as an LLM ---


def test_reply_text_is_the_argmax_letter_json_and_scores_like_an_llm_reply():
    probs = {"latent_risk": {"A": 0.1, "B": 0.0, "C": 0.0, "D": 0.8, "E": 0.1},
             "latent_severity": {"A": 0.2, "B": 0.2, "C": 0.5, "D": 0.1},
             "effect": {"A": 0.0, "B": 0.9, "C": 0.1, "D": 0.0},
             "activated_risk": {"A": 0.6, "B": 0.4, "C": 0.0, "D": 0.0}}
    text = reply_text(probs)
    assert json.loads(text) == {response_key("latent_risk"): "D", response_key("latent_severity"): "C",
                                response_key("effect"): "B", response_key("activated_risk"): "A"}
    gold = {"latent_risk": "D", "latent_severity": "C", "effect": "B", "activated_risk": "A"}
    outcome = score_item(text, gold)
    assert outcome["parse_ok"] and set(outcome["parse"].values()) == {"ok"}
    assert score({"x": gold}, {"x": text})["latent_risk_accuracy"] == 1.0


def test_unanswered_question_is_left_out_of_the_reply():
    text = reply_text({"latent_risk": {"A": 1.0}, "latent_severity": None})
    assert json.loads(text) == {response_key("latent_risk"): "A"}


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
