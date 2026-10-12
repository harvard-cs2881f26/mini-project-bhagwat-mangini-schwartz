"""Text benchmark (Fig. 4): loader, request and scorer. Scorer tests use hand-made replies."""

import json

import pytest

from asimov.bench import get_benchmark
from asimov.bench.parts import TextPart
from asimov.bench.text import PROMPT_NAME, build_request, prompt_version
from asimov.bench.text_score import OPTIONS, QUESTIONS, parse_letters, score
from asimov.data.rows import read_rows
from asimov.data.text import load_text_answers, load_text_items

GOLD = {
    "a": {"latent_risk": "D", "latent_severity": "C", "effect": "B", "activated_risk": "A"},
    "b": {"latent_risk": "A", "latent_severity": "D", "effect": "C", "activated_risk": "D"},
}


def reply(letters: dict[str, str], quote: str = '"') -> str:
    body = ", ".join(f"{quote}{q}_response{quote}: {quote}{v}{quote}" for q, v in letters.items())
    return "{" + body + "}"


# --- loader and request ---


def test_items_carry_only_id_and_prompt():
    rows = read_rows("text")
    items = load_text_items()
    assert [i.id for i in items] == [r["id"] for r in rows]
    assert all(i.prompt == r["prompt"] for i, r in zip(items, rows, strict=True))


def test_answers_are_four_letters_per_item():
    answers = load_text_answers()
    assert set(answers) == {r["id"] for r in read_rows("text")}
    for letters in answers.values():
        assert set(letters) == set(QUESTIONS)
        assert all(letters[q] in OPTIONS[q] for q in QUESTIONS)


def test_request_is_the_paper_prompt_verbatim():
    for item in load_text_items():
        assert build_request(item, PROMPT_NAME) == [TextPart(item.prompt)]


def test_prompt_version_names_dataset_and_prompt():
    assert prompt_version(PROMPT_NAME) == "asimov_v2_injuries/0.1.0|text_verbatim"
    with pytest.raises(ValueError, match="text_verbatim"):
        prompt_version("text_v2")


def test_registered():
    assert get_benchmark("text").name == "text"


# --- reading one answer letter ---


@pytest.mark.parametrize(
    "raw,letter",
    [
        ("D", "D"),
        (" d ", "D"),
        ("D. Concussions", "D"),
        ("(B)", "B"),
        ("C) Moderate", "C"),
        ("A: No injury", "A"),
        ("B Minor Injury", "B"),
    ],
)
def test_letter_forms_accepted(raw, letter):
    assert parse_letters({"latent_risk_response": raw})["latent_risk"] == letter


@pytest.mark.parametrize("raw", ["", "An injury", "Concussions", "F", None, 3, ["D"]])
def test_letter_forms_rejected(raw):
    assert parse_letters({"latent_risk_response": raw})["latent_risk"] is None


def test_letter_outside_the_question_options_is_unanswered():
    assert parse_letters({"latent_risk_response": "E"})["latent_risk"] == "E"  # Q1 has 5 options
    assert parse_letters({"latent_severity_response": "E"})["latent_severity"] is None  # Q2-Q4 have 4


# --- scoring ---


def test_all_correct():
    s = score(GOLD, {k: reply(v) for k, v in GOLD.items()})
    assert s["n"] == 2 and s["parse_failure_rate"] == 0
    assert all(s[f"{q}_accuracy"] == 1 for q in QUESTIONS)
    assert s["per_item"]["a"]["latent_risk"] == {"pred": "D", "gold": "D", "correct": True}
    assert s["per_item"]["a"]["parse_ok"] is True


def test_all_wrong():
    wrong = {"latent_risk": "E", "latent_severity": "A", "effect": "D", "activated_risk": "B"}
    s = score(GOLD, {"a": reply(wrong), "b": reply(wrong)})
    assert all(s[f"{q}_accuracy"] == 0 for q in QUESTIONS)
    assert s["parse_failure_rate"] == 0


def test_single_quoted_json_parses():
    s = score(GOLD, {"a": reply(GOLD["a"], quote="'")})
    assert s["latent_risk_accuracy"] == 1 and s["parse_failure_rate"] == 0


def test_paper_format_with_stray_brace_and_rationales():
    # the paper's format spec has a '}' after activated_risk_response; models may copy it
    text = (
        '```json\n{"latent_risk_response": "D", "latent_risk_rationale": "Head hits the floor.",\n'
        ' "latent_severity_response": "C", "latent_severity_rationale": "Mild concussion.",\n'
        ' "effect_response": "B", "effect_rationale": "Guiding down lowers the risk.",\n'
        ' "activated_risk_response": "A"}, "activated_risk_rationale": "Cushion prevents impact."}\n```'
    )
    s = score(GOLD, {"a": text})
    assert all(s[f"{q}_accuracy"] == 1 for q in QUESTIONS)


def test_missing_key_counts_wrong_and_as_parse_failure():
    partial = {k: v for k, v in GOLD["a"].items() if k != "effect"}
    s = score(GOLD, {"a": reply(partial)})
    assert s["effect_accuracy"] == 0 and s["effect_unanswered_rate"] == 1
    assert s["latent_risk_accuracy"] == 1
    assert s["parse_failure_rate"] == 1 and s["per_item"]["a"]["parse_ok"] is False
    assert s["per_item"]["a"]["effect"] == {"pred": None, "gold": "B", "correct": False}


def test_garbage_is_a_parse_failure():
    s = score(GOLD, {"a": "I can't help with that.", "b": reply(GOLD["b"])})
    assert s["parse_failure_rate"] == 0.5
    assert all(s[f"{q}_accuracy"] == 0.5 for q in QUESTIONS)
    assert all(s[f"{q}_unanswered_rate"] == 0.5 for q in QUESTIONS)


def test_severe_vs_not_and_severe_recall():
    # gold severity: a=C (not severe), b=D (severe); gold activated: a=A, b=D
    preds = {
        "a": {**GOLD["a"], "latent_severity": "B", "activated_risk": "D"},  # sev: not/not ok; act: wrong side
        "b": {**GOLD["b"], "latent_severity": "D", "activated_risk": "C"},
    }  # sev: severe ok; act: missed severe
    s = score(GOLD, {k: reply(v) for k, v in preds.items()})
    assert s["latent_severity_accuracy"] == 0.5
    assert s["latent_severity_severe_binary_accuracy"] == 1 and s["latent_severity_severe_recall"] == 1
    assert s["activated_risk_severe_binary_accuracy"] == 0 and s["activated_risk_severe_recall"] == 0
    assert s["latent_severity_n_severe"] == 1


def test_severe_recall_is_none_without_severe_items():
    s = score({"a": GOLD["a"]}, {"a": reply(GOLD["a"])})
    assert s["latent_severity_severe_recall"] is None


def test_per_item_is_json_serializable_and_scores_only_replied_ids():
    s = score(GOLD, {"a": reply(GOLD["a"])})
    assert s["n"] == 1 and set(s["per_item"]) == {"a"}
    json.dumps(s)


def test_no_replies_raises():
    with pytest.raises(ValueError):
        score(GOLD, {})
