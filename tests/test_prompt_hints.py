"""Text prompt variants with per-item hints from a saved run (experiments/<name>/variant.yaml).

The two Part 2 prompt experiments (EXPLORATORY, WORKLOG 2026-10-14) take Jev's saved picks from run
part2_jev: `text_jev_hints` appends all 4 picks after the verbatim prompt; `text_q2_q4_jev` removes Q1
and states Jev's Q1 pick as the injury Q2-Q4 refer to. These tests read Jev's saved replies on their own
(plain JSON, no scorer) and check every request carries exactly Jev's picks and nothing from the key.
"""

import json
import shutil

import pytest
import yaml

from asimov.bench import text
from asimov.bench.text import PROMPT_NAME, PROMPTS, build_request, is_decision_prompt, is_hint_variant, prompt_version
from asimov.bench.text_score import QUESTIONS, response_key, score_item
from asimov.data.text import TextItem, load_text_answers, load_text_items, parse_options, parse_prompt
from asimov.runner import ROOT

HINTS, Q2_Q4 = "text_jev_hints", "text_q2_q4_jev"
VARIANTS = (HINTS, Q2_Q4)


def jev_rows() -> list[dict]:
    ids = yaml.safe_load((ROOT / "results" / "run_ids.yaml").read_text())
    folder = ROOT / "results" / f"{ids['part2_jev']}_part2_jev" / "jev-latest"
    key = json.loads((folder / "summary.json").read_text())["key"]
    rows = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
    return [r for r in rows if r["key"] == key]


def jev_picks() -> dict[str, dict[str, str]]:
    """Jev's saved letters per item, read straight from its (always clean) JSON replies."""
    return {r["id"]: {q: json.loads(r["text"])[response_key(q)] for q in QUESTIONS} for r in jev_rows()}


def template(name: str) -> str:
    return (ROOT / "experiments" / name / "template.txt").read_text().strip()


def request_text(item: TextItem, name: str) -> str:
    (part,) = build_request(item, name)
    return part.text  # type: ignore[union-attr]


def test_variants_are_text_prompts_with_their_own_version():
    versions = {prompt_version(p) for p in (PROMPT_NAME, *VARIANTS)}
    assert len(versions) == 3
    for name in VARIANTS:
        assert is_hint_variant(name) and name in PROMPTS and not is_decision_prompt(name)
        assert f"|{name}|" in prompt_version(name)
    assert not is_hint_variant(PROMPT_NAME) and not is_hint_variant("text_letter_only")


def test_version_changes_when_the_template_changes(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "experiments" / HINTS, tmp_path / HINTS)
    monkeypatch.setattr(text, "EXPERIMENTS", tmp_path)
    text.variant.cache_clear()
    try:
        before = prompt_version(HINTS)
        (tmp_path / HINTS / "template.txt").write_text("Other wording {latent_risk} {latent_severity} {effect} {activated_risk}")
        text.variant.cache_clear()
        assert prompt_version(HINTS) != before
    finally:
        text.variant.cache_clear()


def test_every_item_has_a_saved_jev_pick():
    picks = jev_picks()
    assert {i.id for i in load_text_items()} <= set(picks)
    assert all(letter for p in picks.values() for letter in p.values())


def test_hint_letters_match_the_scorers_reading_of_jev():
    answers, replies = load_text_answers(), {r["id"]: r["text"] for r in jev_rows()}
    v = text.variant(HINTS)
    assert v is not None
    source = v.source
    for item in load_text_items():
        scored = score_item(replies[item.id], answers[item.id], parse_options(item.prompt))
        assert text.item_hints(item, source) == {q: scored[q]["pred"] for q in QUESTIONS}, item.id


def test_hints_request_is_the_verbatim_prompt_plus_exactly_jevs_picks():
    picks = jev_picks()
    for item in load_text_items():
        sent = request_text(item, HINTS)
        assert sent.startswith(item.prompt + "\n\n"), item.id
        options = parse_options(item.prompt)
        shown = {q: f"{picks[item.id][q]}. {options[q][picks[item.id][q]]}" for q in QUESTIONS}
        assert sent[len(item.prompt) + 2 :] == template(HINTS).format(**shown), item.id


def test_hints_add_no_key_answer_jev_did_not_give():
    picks, answers = jev_picks(), load_text_answers()
    for item in load_text_items():
        added = request_text(item, HINTS)[len(item.prompt) :].splitlines()
        options = parse_options(item.prompt)
        for k, q in enumerate(QUESTIONS, 1):
            gold = answers[item.id][q]
            if gold != picks[item.id][q]:  # Q2's option A is a prefix of Q4's, so compare whole lines
                assert f"Q{k}: {gold}. {options[q][gold]}" not in added, (item.id, q)
            assert sum(line.startswith(f"Q{k}: ") for line in added) == 1, (item.id, q)


def test_q2_q4_request_has_no_q1_and_keeps_q2_to_q4_verbatim():
    picks, answers = jev_picks(), load_text_answers()
    for item in load_text_items():
        sent, parsed = request_text(item, Q2_Q4), parse_prompt(item.prompt)
        q1, rest = parsed.questions[0], parsed.questions[1:]
        assert q1.text not in sent and "Q1. (Latent Risk)" not in sent, item.id
        assert not any(f"\n{letter}. {option}" in sent for letter, option in q1.options.items()), item.id
        assert "latent_risk_response" not in sent and "latent_risk_rationale" not in sent, item.id
        for q in rest:
            assert q.text in sent and all(f"\n{letter}. {option}" in sent for letter, option in q.options.items()), item.id
            assert f"'{response_key(q.key)}'" in sent, item.id
        assert parsed.scenario in sent and parsed.instruction in sent and sent.endswith(item.prompt[-200:]), item.id
        jev, gold = picks[item.id]["latent_risk"], answers[item.id]["latent_risk"]
        assert template(Q2_Q4).format(latent_risk=q1.options[jev]) in sent, item.id
        if gold != jev:
            assert f": {q1.options[gold]}." not in sent, item.id  # the key's injury is never stated


@pytest.mark.full_data
def test_all_319_items_build_for_both_variants():
    for name in VARIANTS:
        assert len([build_request(i, name) for i in load_text_items()]) == 319


def test_item_without_a_saved_pick_fails_loudly():
    item = load_text_items()[0]
    with pytest.raises(ValueError, match="no saved reply"):
        build_request(TextItem(id="text_9999", prompt=item.prompt), HINTS)
