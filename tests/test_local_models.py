"""Local decision models (Part 2, Step C): Qwen option probabilities, Qwen generation, NLI.

Only the pure parts are tested here, offline, with fake tokenizers and fake logits: rendering a
DecisionPart into text, the one-token letter check, scores -> option probabilities, the NLI
entailment column, truncation detection, configs and prompt versions. No model is downloaded
and torch is never needed (the one torch test is skipped without it).
"""

import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence

import pytest
import yaml

from asimov.bench.parts import DecisionPart, TextPart
from asimov.bench.text import (
    DECISION_PROMPTS,
    LOCAL_DECISION_PROMPTS,
    build_request,
    is_local_prompt,
    parse_decision_prompt,
    prompt_version,
    wording,
)
from asimov.bench.text_score import QUESTIONS, response_key
from asimov.config import ModelSpec, RunConfig
from asimov.data.text import load_text_items
from asimov.models import request_settings
from asimov.models.base import ModelResponse
from asimov.models.local_hf import LocalHFClient, check_backend, check_params, generation_finish
from asimov.models.local_text import (
    answer_token,
    entailment_index,
    hypotheses,
    letter_token_ids,
    option_letters,
    option_probs,
    plain_prompt,
    question_text,
    softmax,
)
from asimov.models.nli import NLIClient
from asimov.models.remote import part_from_dict, part_to_dict, serve_complete, serve_decide
from asimov.models.reply_info import finish
from asimov.runner import ROOT

QWEN_BASE = {"repo": "Qwen/Qwen3.5-4B-Base", "revision": "1001bb4d826a52d1f399e183466143f4da7b741b"}


@pytest.fixture
def part() -> DecisionPart:
    [p] = build_request(load_text_items()[0], "decision_short_local")
    assert isinstance(p, DecisionPart)
    return p


# --- prompt names and versions ---


def test_local_prompts_build_the_same_request_as_their_hosted_names():
    item = load_text_items()[0]
    for name in DECISION_PROMPTS:
        local = name + "_local"
        assert local in LOCAL_DECISION_PROMPTS and is_local_prompt(local) and not is_local_prompt(name)
        assert parse_decision_prompt(local) == parse_decision_prompt(name)
        assert build_request(item, local) == build_request(item, name)
    assert not is_local_prompt("text_verbatim")


def test_local_wording_is_in_the_local_prompt_versions_only(monkeypatch):
    names = (*DECISION_PROMPTS, *LOCAL_DECISION_PROMPTS)
    before = {p: prompt_version(p) for p in names}
    assert len(set(before.values())) == len(names)
    edited = {**wording(), "local": {**wording()["local"], "answer_cue": "\nEDITED:"}}
    monkeypatch.setattr("asimov.bench.text.wording", lambda: edited)
    for p in names:
        assert (prompt_version(p) != before[p]) == is_local_prompt(p), p


# --- rendering a DecisionPart as text (wording only from the `local` section) ---


def test_question_text_has_state_instructions_and_lettered_options_in_sent_order(part):
    w = wording()["local"]
    for q in part.questions:
        text = question_text(part.state, q)
        assert text.startswith(part.state) and q.instructions in text
        lines = [w["option"].format(letter=letter, description=d) for letter, (_, d) in zip("ABCDE", q.options, strict=False)]
        assert "\n".join(lines) in text
        assert option_letters(q) == tuple("ABCDE"[: len(q.options)])
        assert plain_prompt(part.state, q) == text + w["answer_cue"]


def test_rendered_text_is_only_the_part_and_the_template(part):
    """Nothing but the state, instructions and option descriptions is filled in (no ids, no answers)."""
    w = wording()["local"]
    q = part.questions[0]
    filled = question_text(part.state, q)
    template = w["question"].format(state="", instructions="", options="")
    rest = filled.replace(part.state, "", 1).replace(q.instructions, "", 1)
    for letter, (_, d) in zip(option_letters(q), q.options, strict=True):
        rest = rest.replace(w["option"].format(letter=letter, description=d), "", 1)
    assert rest.replace("\n", "") == template.replace("\n", "")


def test_nli_hypotheses_one_per_option_from_the_template(part):
    for q in part.questions:
        hyps = hypotheses(q)
        template = wording()["local"]["nli_hypothesis"][q.id]
        assert hyps == [template.format(description=d) for _, d in q.options]


# --- letters must be single tokens, right after the prompt ---


TOKEN = re.compile(r" ?[A-Z]|.", re.S)  # " A" and "A" are one token each, like Qwen's byte-level BPE


def encoder(pattern: re.Pattern = TOKEN):
    vocab: dict[str, int] = {}
    return lambda s: [vocab.setdefault(t, len(vocab)) for t in pattern.findall(s)]


def test_answer_token_has_a_space_only_after_a_non_space():
    assert answer_token("ANSWER:", "A") == " A"
    assert answer_token("<think>\n\n</think>\n\n", "A") == "A"


def test_letter_token_ids_are_one_token_each_and_distinct():
    encode = encoder()
    ids = letter_token_ids(encode, "Q?\nANSWER:", "ABCDE")
    assert ids == [encode(" " + letter)[0] for letter in "ABCDE"] and len(set(ids)) == 5
    assert letter_token_ids(encode, "</think>\n\n", "AB") == [encode("A")[0], encode("B")[0]]


def test_letter_split_in_two_tokens_fails_loudly():
    with pytest.raises(ValueError, match="not one token"):
        letter_token_ids(encoder(re.compile(r".", re.S)), "ANSWER:", "AB")  # " A" = two tokens


def test_letter_merging_with_the_prompt_end_fails_loudly():
    with pytest.raises(ValueError, match="not one token"):
        letter_token_ids(encoder(re.compile(r": [A-Z]|.", re.S)), "ANSWER:", "A")  # ": A" merges


# --- scores -> option probabilities ---


def test_softmax_sums_to_one_and_is_stable():
    p = softmax([1000.0, 1000.0, 0.0])
    assert p[0] == pytest.approx(0.5) and sum(p) == pytest.approx(1) and p[2] == pytest.approx(0)


def test_option_probs_map_scores_to_values_in_sent_order(part):
    q = part.questions[2]  # effect: eliminates, reduces, no_change, increases
    probs = option_probs(q, [0.0, 2.0, 0.0, 0.0])
    assert list(probs) == [v for v, _ in q.options] and max(probs, key=lambda v: probs[v]) == "reduces"
    assert sum(probs.values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        option_probs(q, [0.0, 1.0])


def test_entailment_index_from_the_label_names():
    assert entailment_index({"entailment": 0, "not_entailment": 1}) == 0
    assert entailment_index({"contradiction": 0, "neutral": 1, "Entailment": 2}) == 2
    with pytest.raises(ValueError, match="entail"):
        entailment_index({"LABEL_0": 0, "LABEL_1": 1})


# --- clients, with the model replaced by fixed scores ---


class FakeTokenizer:
    def __init__(self):
        self.encode_text = encoder()

    def encode(self, text, add_special_tokens=False):
        return self.encode_text(text)

    def decode(self, ids):
        return "?"


def fake_local_client(scores: Mapping[str, Sequence[float]], template: str = "plain") -> LocalHFClient:
    """A logprob client whose next-token scores per question id are fixed; no model loaded."""
    client = LocalHFClient.__new__(LocalHFClient)
    client.model, client.lock, client.backend = "qwen3.5-4b-base", threading.Lock(), "local"
    client.repo, client.revision, client.mode, client.template = *QWEN_BASE.values(), "logprob", template
    client.tokenizer = FakeTokenizer()
    asked = iter(QUESTIONS)

    def letter_logits(ids, letter_ids):
        return scores[next(asked)], {"top_token": " B", "letter_mass": 0.9}

    client.letter_logits = letter_logits  # pyright: ignore[reportAttributeAccessIssue]
    return client


def test_logprob_client_reply_is_argmax_letters_with_probs(part):
    scores = {
        "latent_risk": [0, 0, 0, 3, 0],
        "latent_severity": [0, 0, 1, 0],
        "effect": [0, 2, 0, 0],
        "activated_risk": [1, 0, 0, 0],
    }
    r = fake_local_client(scores).generate([part])
    assert json.loads(r.text) == {
        response_key("latent_risk"): "D",
        response_key("latent_severity"): "C",
        response_key("effect"): "B",
        response_key("activated_risk"): "A",
    }
    assert all(sum(d.values()) == pytest.approx(1) for d in r.probs.values())
    assert r.model == "Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b"
    assert r.usage["output_tokens"] == 0 and r.usage["input_tokens"] > 0
    assert r.raw["questions"]["latent_risk"]["letter_logits"] == [0, 0, 0, 3, 0]
    assert finish("local_hf", r.raw) == "complete"


def test_logprob_client_cumulative_makes_one_pass_per_question():
    [part] = build_request(load_text_items()[0], "decision_short_cumulative_local")
    scores = {q: [1.0, 0, 0, 0, 0] if q == "latent_risk" else [1.0, 0, 0, 0] for q in QUESTIONS}
    r = fake_local_client(scores).generate([part])
    assert r.raw["mode"] == "cumulative" and len(r.raw["calls"]) == 4
    assert set(json.loads(r.text).values()) == {"A"}


def test_nli_client_softmaxes_entailment_over_options(part):
    client = NLIClient.__new__(NLIClient)
    client.model, client.lock, client.backend = "deberta-v3-large-zeroshot-v2.0", threading.Lock(), "local"
    client.repo, client.revision = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "cf44676c"
    sent = []

    def entailment_logits(premise, hyps):
        sent.append((premise, hyps))
        return [float(i == 1) * 3 for i in range(len(hyps))], 100 * len(hyps)

    client.entailment_logits = entailment_logits
    r = client.generate([part])
    assert [premise for premise, _ in sent] == [part.state] * 4
    assert [hyps for _, hyps in sent] == [hypotheses(q) for q in part.questions]
    assert set(json.loads(r.text).values()) == {"B"}
    assert all(sum(d.values()) == pytest.approx(1) for d in r.probs.values())
    assert r.usage == {"input_tokens": 1700, "output_tokens": 0}  # 5 + 4 + 4 + 4 pairs of 100 tokens


# --- generation: greedy, thinking off, truncation read from the saved raw ---


def test_generation_finish_is_length_only_when_cut_at_the_limit():
    assert generation_finish(n_new=1024, max_new=1024, ended_with_eos=False) == "length"
    assert generation_finish(n_new=1024, max_new=1024, ended_with_eos=True) == "stop"
    assert generation_finish(n_new=200, max_new=1024, ended_with_eos=True) == "stop"


def test_finish_reads_local_truncation():
    assert finish("local_hf", {"finish_reason": "length"}) == "truncated"
    assert finish("local_hf", {"finish_reason": "stop"}) == "complete"
    assert finish("nli", {"questions": {}}) == "complete"


# --- params, configs ---


def test_params_are_checked():
    assert check_params({**QWEN_BASE, "mode": "logprob", "template": "plain"}) == (*QWEN_BASE.values(), "logprob", "plain")
    for bad in (
        {**QWEN_BASE, "mode": "sample", "template": "plain"},
        {**QWEN_BASE, "mode": "logprob", "template": "markdown"},
        {**QWEN_BASE, "mode": "generate", "template": "plain"},  # generation is always the chat template
        {"repo": QWEN_BASE["repo"], "mode": "logprob", "template": "plain"},  # no pinned revision
        {**QWEN_BASE, "mode": "logprob", "template": "plain", "temperature": 1},
    ):
        with pytest.raises(ValueError):
            check_params(bad)


def test_local_providers_take_params_but_not_service_fields():
    ModelSpec(
        name="q", provider="local_hf", model="qwen3.5-4b-base", params={**QWEN_BASE, "mode": "logprob", "template": "plain"}
    )
    ModelSpec(name="n", provider="nli", model="deberta", params={"repo": "x/y", "revision": "abc"})
    with pytest.raises(ValueError, match="systemone"):
        ModelSpec(name="q", provider="local_hf", model="q", base_url="http://localhost", params={**QWEN_BASE})
    assert request_settings("local_hf") == request_settings("nli") == frozenset()


def run_config(provider: str, prompt: str, mode: str = "logprob") -> dict:
    params = {"repo": "x/y", "revision": "abc"} | ({"mode": mode, "template": "chat"} if provider == "local_hf" else {})
    return {
        "run": "t",
        "benchmark": "text",
        "prompt": prompt,
        "thinking": "min",
        "models": [{"name": "m", "provider": provider, "model": "m", "params": params}],
    }


def test_local_decision_models_need_a_local_prompt():
    RunConfig.model_validate(run_config("local_hf", "decision_short_local"))
    RunConfig.model_validate(run_config("nli", "decision_full_cumulative_local"))
    RunConfig.model_validate(run_config("local_hf", "text_verbatim", mode="generate"))
    for provider, prompt, mode in (
        ("local_hf", "decision_short", "logprob"),
        ("nli", "decision_short", "logprob"),
        ("local_hf", "text_verbatim", "logprob"),
        ("local_hf", "decision_short_local", "generate"),
    ):
        with pytest.raises(ValueError):
            RunConfig.model_validate(run_config(provider, prompt, mode))


LOCAL_CONFIGS = ("part2_qwen_base_logprob", "part2_qwen_logprob", "part2_qwen_gen", "part2_nli")


@pytest.mark.parametrize("name", LOCAL_CONFIGS)
def test_local_configs_load_with_pinned_revisions(name):
    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    RunConfig.model_validate(cfg)
    assert cfg["run"] == name
    [spec] = cfg["models"]
    assert re.fullmatch(r"[0-9a-f]{40}", spec["params"]["revision"])
    assert "/" not in spec["model"]  # results/<run>/<model>/ is one folder


# --- torch stays out of CI ---


def test_importing_the_local_clients_does_not_import_torch():
    code = "import sys, asimov.models.local_hf, asimov.models.nli; assert 'torch' not in sys.modules"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr


def test_letter_logits_from_a_logits_vector():
    torch = pytest.importorskip("torch")
    from asimov.models.local_hf import letter_stats

    logits = torch.tensor([0.0, 5.0, 1.0, 1.0, -2.0])
    scores, info = letter_stats(logits, [2, 3], decode=lambda ids: f"tok{ids[0]}")
    assert scores == [1.0, 1.0]
    assert info["top_token"] == "tok1"
    assert info["letter_mass"] == pytest.approx(float(torch.softmax(logits, -1)[[2, 3]].sum()))


# --- scripts/agreement.py: same answer in two runs, per question ---


def test_agreement_counts_shared_items_and_unanswered():
    sys.path.insert(0, str(ROOT / "scripts"))
    from agreement import agreement  # pyright: ignore[reportMissingImports]

    def item(*preds):
        return {q: {"pred": p} for q, p in zip(QUESTIONS, preds, strict=True)}

    a = {"x": item("A", "B", "C", None), "y": item("A", "A", "A", "A"), "only_a": item("A", "A", "A", "A")}
    b = {"x": item("A", "C", "C", None), "y": item("B", "A", "A", "A")}
    out = agreement(a, b)
    assert out["latent_risk"] == {"n": 2, "agree": 1, "unanswered_a": 0, "unanswered_b": 0}
    assert out["effect"]["agree"] == 2
    assert out["activated_risk"] == {"n": 2, "agree": 1, "unanswered_a": 1, "unanswered_b": 1}  # both unanswered: no


# --- Modal backend: the same client code, run remotely (models/remote.py), with a fake remote ---


class FakeMethod:
    def __init__(self, fn):
        self.remote = fn


class FakeRemote:
    """Stands in for the deployed Modal class: runs the server-side handlers on a local fake client."""

    def __init__(self, client):
        info = {"backend": "modal", "device": "cuda", "gpu": "NVIDIA L40S", "load_s": 30.0}
        self.decide = FakeMethod(lambda part: serve_decide(client, part, info))
        self.complete = FakeMethod(lambda text: serve_complete(client, text, info))


def modal_client(remote, mode: str = "logprob", template: str = "plain") -> LocalHFClient:
    client = LocalHFClient.__new__(LocalHFClient)
    client.model, client.lock, client.backend, client.remote = "qwen3.5-4b-base", threading.Lock(), "modal", remote
    client.repo, client.revision, client.mode, client.template = *QWEN_BASE.values(), mode, template
    return client


SCORES = {"latent_risk": [0, 0, 0, 3, 0], "latent_severity": [0, 0, 1, 0], "effect": [0, 2, 0, 0], "activated_risk": [1, 0, 0, 0]}


def test_decision_part_crosses_as_a_dict_unchanged():
    item = load_text_items()[0]
    for name in ("decision_short_local", "decision_full_letters_cumulative_local"):
        [p] = build_request(item, name)
        assert isinstance(p, DecisionPart)
        assert part_from_dict(json.loads(json.dumps(part_to_dict(p)))) == p


def test_modal_backend_gives_the_same_reply_and_records_where_it_ran(part):
    here = fake_local_client(SCORES).generate([part])
    there = modal_client(FakeRemote(fake_local_client(SCORES))).generate([part])
    assert (there.text, there.probs, there.usage, there.model) == (here.text, here.probs, here.usage, here.model)
    remote = there.raw["remote"]
    assert remote["backend"] == "modal" and remote["gpu"] == "NVIDIA L40S" and remote["compute_s"] >= 0
    assert {k: v for k, v in there.raw.items() if k != "remote"} == here.raw


def test_modal_reply_made_with_other_wording_raises(part):
    remote = FakeRemote(fake_local_client(SCORES))
    real = remote.decide.remote
    remote.decide = FakeMethod(lambda p: {**real(p), "local_wording": "000000000000"})
    with pytest.raises(RuntimeError, match="redeploy"):
        modal_client(remote).generate([part])


class Generator:
    def complete(self, text):
        raw = {"finish_reason": "length", "max_new_tokens": 1024}
        usage = {"input_tokens": 5, "output_tokens": 1024}
        return ModelResponse(text="{}", model="Qwen/Qwen3.5-4B@x", usage=usage, latency_s=2.5, raw=raw)


def test_modal_generation_keeps_finish_and_records_compute_time():
    r = modal_client(FakeRemote(Generator()), mode="generate", template="chat").generate([TextPart("prompt")])
    assert r.text == "{}" and r.usage["output_tokens"] == 1024
    assert r.raw["remote"]["compute_s"] == 2.5 and finish("local_hf", r.raw) == "truncated"


def test_backend_param_is_checked_and_optional():
    assert check_backend({}) == "local" and check_backend({"backend": "modal"}) == "modal"
    with pytest.raises(ValueError, match="backend"):
        check_backend({"backend": "gpu"})
    assert check_params({**QWEN_BASE, "mode": "logprob", "template": "plain", "backend": "modal"})[2] == "logprob"


@pytest.mark.parametrize("name", LOCAL_CONFIGS)
def test_local_configs_run_on_modal(name):
    [spec] = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())["models"]
    assert spec["params"]["backend"] == "modal"


def test_importing_the_remote_path_imports_neither_modal_nor_torch():
    code = (
        "import sys, asimov.models.remote, asimov.models.local_hf, asimov.models.nli; "
        "assert 'modal' not in sys.modules and 'torch' not in sys.modules"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
