"""One-shot variant of the base-model option-probability prompt (decision_*_oneshot_local).

One made-up worked example per question, rendered like the real question and answered, goes
before the real item (plain template only). Checked offline: names and versions, rendering, the
example is invented (not in the dataset) and never carries an item's answer or id.
"""

import json
import re

import pytest
import yaml

from asimov.bench.parts import DecisionPart
from asimov.bench.text import (
    DECISION_PROMPTS,
    LOCAL_DECISION_PROMPTS,
    ONESHOT_PROMPTS,
    build_request,
    is_local_prompt,
    is_oneshot_prompt,
    parse_decision_prompt,
    prompt_version,
    wording,
)
from asimov.config import RunConfig
from asimov.data.rows import read_rows
from asimov.data.text import load_text_items, parse_prompt
from asimov.models.local_text import answer_token, letter_token_ids, local_wording, plain_prompt, question_text
from asimov.models.remote import part_from_dict, part_to_dict
from asimov.runner import ROOT

NAME = "decision_short_oneshot_local"
TOKEN = re.compile(r" ?[A-Z]|.", re.S)


def oneshot_part(item=None) -> DecisionPart:
    [p] = build_request(item or load_text_items()[0], NAME)
    assert isinstance(p, DecisionPart)
    return p


def test_oneshot_names_parse_like_their_base_names():
    for name in DECISION_PROMPTS:
        oneshot = name + "_oneshot_local"
        assert oneshot in ONESHOT_PROMPTS and is_oneshot_prompt(oneshot) and is_local_prompt(oneshot)
        assert parse_decision_prompt(oneshot) == parse_decision_prompt(name)
        assert not is_oneshot_prompt(name) and not is_oneshot_prompt(name + "_local")
    with pytest.raises(ValueError):
        parse_decision_prompt("decision_short_oneshot")  # one-shot is a local-model prompt only


def test_oneshot_wording_moves_only_the_oneshot_versions(monkeypatch):
    names = (*DECISION_PROMPTS, *LOCAL_DECISION_PROMPTS, *ONESHOT_PROMPTS)
    before = {p: prompt_version(p) for p in names}
    assert len(set(before.values())) == len(names)
    edited = {**wording(), "oneshot": {**wording()["oneshot"], "answers": {**wording()["oneshot"]["answers"], "effect": "C"}}}
    monkeypatch.setattr("asimov.bench.text.wording", lambda: edited)
    for p in names:
        assert (prompt_version(p) != before[p]) == is_oneshot_prompt(p), p


def test_oneshot_request_is_the_local_request_plus_one_example_per_question():
    item = load_text_items()[0]
    [plain] = build_request(item, "decision_short_local")
    assert isinstance(plain, DecisionPart)
    part = oneshot_part(item)
    assert part.state == plain.state
    w = wording()["oneshot"]
    for q, base in zip(part.questions, plain.questions, strict=True):
        assert (q.id, q.instructions, q.options, q.letters) == (base.id, base.instructions, base.options, base.letters)
        ex = q.example
        assert ex is not None and base.example is None
        assert ex.state == wording()["state"].format(**w["example"])
        expected = w["latent_risk_options"] if q.id == "latent_risk" else [d for _, d in q.options]
        assert list(ex.options) == list(expected) and ex.answer == w["answers"][q.id]


def test_plain_prompt_puts_the_answered_example_before_the_real_item():
    part = oneshot_part()
    cue = local_wording()["answer_cue"]
    for q in part.questions:
        text = plain_prompt(part.state, q)
        assert text.endswith(question_text(part.state, q) + cue)
        example = text[: -len(question_text(part.state, q) + cue)]
        assert q.example is not None and example.startswith(q.example.state)
        assert example.endswith(cue + answer_token(cue, q.example.answer) + "\n\n")
        assert part.state not in example


def test_oneshot_prompt_letters_are_single_tokens_after_the_cue():
    vocab: dict[str, int] = {}

    def encode(s):
        return [vocab.setdefault(t, len(vocab)) for t in TOKEN.findall(s)]

    part = oneshot_part()
    for q in part.questions:
        assert len(letter_token_ids(encode, plain_prompt(part.state, q), "ABCDE"[: len(q.options)])) == len(q.options)


def test_example_is_invented_and_never_carries_an_items_answer_or_id():
    w = wording()["oneshot"]["example"]
    words = set(re.findall(r"[a-z]+", (w["scenario"] + " " + w["instruction"]).lower()))
    raw = {r["id"]: r["answer_raw"] for r in read_rows("text")}
    for item in load_text_items():
        p = parse_prompt(item.prompt)
        assert w["scenario"] not in item.prompt and w["instruction"] not in item.prompt
        other = set(re.findall(r"[a-z]+", (p.scenario + " " + p.instruction).lower()))
        assert len(words & other) / len(words | other) < 0.3, item.id  # not a near-duplicate
        for q in oneshot_part(item).questions:
            assert q.example is not None
            text = q.example.state + "\n".join(q.example.options) + q.example.answer
            assert raw[item.id] not in text and item.id not in text


def test_oneshot_part_crosses_as_a_dict_unchanged():
    p = oneshot_part()
    assert part_from_dict(json.loads(json.dumps(part_to_dict(p)))) == p


def config(provider: str, prompt: str, template: str = "plain") -> dict:
    params = {"repo": "x/y", "revision": "abc"} | ({"mode": "logprob", "template": template} if provider == "local_hf" else {})
    return {"run": "t", "benchmark": "text", "prompt": prompt, "thinking": "min",
            "models": [{"name": "m", "provider": provider, "model": "m", "params": params}]}  # fmt: skip


def test_oneshot_is_for_the_plain_template_only():
    RunConfig.model_validate(config("local_hf", NAME))
    for provider, template in (("local_hf", "chat"), ("nli", "plain")):
        with pytest.raises(ValueError, match="one-shot"):
            RunConfig.model_validate(config(provider, NAME, template))


def test_oneshot_config():
    cfg = yaml.safe_load((ROOT / "configs" / "part2_qwen_base_logprob_oneshot.yaml").read_text())
    RunConfig.model_validate(cfg)
    base = yaml.safe_load((ROOT / "configs" / "part2_qwen_base_logprob.yaml").read_text())
    assert cfg["run"] == "part2_qwen_base_logprob_oneshot" and cfg["prompt"] == NAME
    assert cfg["models"][0]["params"] == base["models"][0]["params"]  # same model and settings, only the prompt differs
