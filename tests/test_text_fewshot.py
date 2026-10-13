"""Few-shot format prompt variant (experiments/text_fewshot_format), EXPLORATORY (WORKLOG 2026-10-14).

The request is prefix.txt (made-up worked examples), a blank line, the dataset's prompt verbatim, a
blank line, then suffix.txt (the format rules). Checked offline: placement, the cache key, the
examples are invented (not in the dataset) and well-formed, and other prompts' versions don't move.
"""

import json
import re

import yaml

from asimov.bench.parts import TextPart
from asimov.bench.text import (
    DATASET_VERSION,
    EXPERIMENTS,
    FEWSHOT_FORMAT,
    LETTER_ONLY,
    PROMPT_NAME,
    build_request,
    prefix,
    prompt_version,
    suffix,
)
from asimov.config import RunConfig
from asimov.data.rows import read_rows
from asimov.data.text import load_text_items, parse_prompt
from asimov.runner import ROOT

FOLDER = EXPERIMENTS / FEWSHOT_FORMAT
RULES = "Q2 and Q4 use only A-D; if Q1 is E, answer A for Q2 and Q4. Keep each rationale to one sentence."
EXAMPLE = re.compile(r"SCENARIO: (.*?) and INSTRUCTION: (.*?)\n", re.S)
KEYS = [f"{q}_{f}" for q in ("latent_risk", "latent_severity", "effect", "activated_risk") for f in ("response", "rationale")]


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def examples() -> list[tuple[str, str, dict]]:
    """(scenario, instruction, answer) for each worked example in prefix.txt."""
    text = prefix(FEWSHOT_FORMAT)
    answers = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    found = EXAMPLE.findall(text)
    assert len(found) == len(answers)
    return [(s, i, a) for (s, i), a in zip(found, answers, strict=True)]


def test_request_is_prefix_then_the_verbatim_prompt_then_the_rules():
    for item in load_text_items():
        [part] = build_request(item, FEWSHOT_FORMAT)
        assert isinstance(part, TextPart)
        assert part.text == prefix(FEWSHOT_FORMAT) + "\n\n" + item.prompt + "\n\n" + suffix(FEWSHOT_FORMAT)
        assert item.prompt in part.text  # the dataset's prompt, unchanged
        assert part.text.index(item.prompt) > part.text.index("Example 3")  # examples come first


def test_rules_sentence_is_the_agreed_wording():
    assert suffix(FEWSHOT_FORMAT) == RULES


def test_prefix_and_suffix_are_in_the_cache_key():
    version = prompt_version(FEWSHOT_FORMAT)
    assert prefix(FEWSHOT_FORMAT) in version and RULES in version
    assert len({version, prompt_version(PROMPT_NAME), prompt_version(LETTER_ONLY)}) == 3


def test_existing_text_prompt_versions_are_unchanged():
    assert prompt_version(PROMPT_NAME) == f"{DATASET_VERSION}|{PROMPT_NAME}"
    assert prompt_version(LETTER_ONLY) == f"{DATASET_VERSION}|{LETTER_ONLY}|{suffix(LETTER_ONLY)}"
    assert prefix(PROMPT_NAME) == prefix(LETTER_ONLY) == ""
    assert build_request(load_text_items()[0], PROMPT_NAME)[0].text == load_text_items()[0].prompt


def test_examples_are_well_formed_and_one_has_q1_e():
    shots = examples()
    assert 2 <= len(shots) <= 3
    for _, _, answer in shots:
        assert list(answer) == KEYS
        assert answer["latent_risk_response"] in "ABCDE"
        assert answer["effect_response"] in "ABCD"
        for q in ("latent_severity", "activated_risk"):
            assert answer[f"{q}_response"] in "ABCD"
            if answer["latent_risk_response"] == "E":
                assert answer[f"{q}_response"] == "A"
        for key in KEYS[1::2]:
            assert len(re.findall(r"[.!?](\s|$)", answer[key])) == 1, key  # one sentence
    assert any(a["latent_risk_response"] == "E" for _, _, a in shots)


def test_examples_are_invented_and_carry_no_items_answer_or_id():
    raw = {r["id"]: r["answer_raw"] for r in read_rows("text")}
    text = prefix(FEWSHOT_FORMAT) + suffix(FEWSHOT_FORMAT)
    for item in load_text_items():
        assert raw[item.id] not in text and item.id not in text
        p = parse_prompt(item.prompt)
        other = words(p.scenario + " " + p.instruction)
        for scenario, instruction, _ in examples():
            assert scenario not in item.prompt and instruction not in item.prompt
            for sentence in re.split(r"(?<=\.) ", scenario + " " + instruction):
                assert sentence not in item.prompt, (item.id, sentence)
            mine = words(scenario + " " + instruction)
            assert len(mine & other) / len(mine | other) < 0.3, item.id  # not a near-duplicate


def test_fewshot_config_changes_only_the_prompt():
    cfg = yaml.safe_load((ROOT / "configs" / "part2_qwen_gen_fewshot.yaml").read_text())
    base = yaml.safe_load((ROOT / "configs" / "part2_qwen_gen_2048.yaml").read_text())
    RunConfig.model_validate(cfg)
    assert cfg["run"] == "part2_qwen_gen_fewshot" and cfg["prompt"] == FEWSHOT_FORMAT
    assert {k: v for k, v in cfg.items() if k not in ("run", "prompt")} == {
        k: v for k, v in base.items() if k not in ("run", "prompt")
    }
