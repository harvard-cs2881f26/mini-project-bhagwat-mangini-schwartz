"""Local Hugging Face causal LMs (Part 2, Step C): Qwen3.5-4B-Base and Qwen3.5-4B, open weights run
by us (on this machine, or on a Modal GPU with the same code).

params (config):
    repo:      Hugging Face repo id, e.g. Qwen/Qwen3.5-4B-Base
    revision:  the commit sha to load, so reruns use the same weights
    mode:      logprob  a decision client: per question, one forward pass on the question rendered
                        as text (models.local_text); the next-token logits of the option letters,
                        softmaxed over those letters only, are the option probabilities. Question
                        modes, ties and the 4-letter reply come from models.decision.
               generate an LLM client: one greedy reply to a text prompt (e.g. text_verbatim),
                        at most MAX_NEW_TOKENS new tokens; raw["finish_reason"] is "length" when cut off.
    template:  plain    the text as is, ending in the answer cue (base model; logprob only)
               chat     the model's chat template with thinking off (enable_thinking=False: an empty
                        think block, so the reply starts with the answer)
    backend:   local    (default) load the model here
               modal    send each request to the deployed Modal app (models/modal_app.py, L40S), which
                        runs this same client with backend local (models/remote.py)
`thinking` doesn't apply: thinking is always off. torch and transformers are imported only when a
model is loaded (requirements-local.txt), so CI never imports them. Weights load in bf16 on CUDA,
else Apple MPS, else CPU. One forward pass at a time (the runner calls from threads).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from asimov.bench.parts import DecisionPart, DecisionQuestion, Part, TextPart
from asimov.models.base import ModelResponse, Thinking, Timer
from asimov.models.decision import Decided, argmax, generate_decision
from asimov.models.local_text import letter_token_ids, option_letters, option_probs, plain_prompt, question_text
from asimov.models.remote import BACKENDS, remote_complete, remote_decide, remote_model

MODES = ("logprob", "generate")
TEMPLATES = ("plain", "chat")
PARAMS = ("repo", "revision", "mode", "template")
MAX_NEW_TOKENS = 1024  # generation cap; the paper's JSON reply with rationales needs a few hundred
# Rendered at load, so a letter that isn't one token fails before any item runs.
LETTER_CHECK = DecisionQuestion(
    id="check", instructions="Which option?", options=tuple((v, v) for v in "vwxyz"), letters=tuple("ABCDE")
)


def check_params(params: dict | None) -> tuple[str, str, str, str]:
    """(repo, revision, mode, template); ValueError naming what's wrong."""
    params = {k: v for k, v in (params or {}).items() if k != "backend"}  # backend: check_backend
    unknown = sorted(set(params) - set(PARAMS))
    missing = [k for k in PARAMS if not params.get(k)]
    if unknown or missing:
        raise ValueError(f"local_hf params: unknown {unknown}, missing {missing}; give exactly {PARAMS}")
    repo, revision, mode, template = (str(params[k]) for k in PARAMS)
    if mode not in MODES or template not in TEMPLATES:
        raise ValueError(f"local_hf params: mode must be one of {MODES}, template one of {TEMPLATES}")
    if mode == "generate" and template != "chat":
        raise ValueError("local_hf generate uses the chat template with thinking off: set template: chat")
    return repo, revision, mode, template


def check_backend(params: dict | None) -> str:
    backend = str((params or {}).get("backend", "local"))
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {backend!r}")
    return backend


def device(torch: Any) -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def generation_finish(n_new: int, max_new: int, ended_with_eos: bool) -> str:
    """ "length" if generation stopped at the token cap without ending its turn, else "stop"."""
    return "length" if n_new >= max_new and not ended_with_eos else "stop"


def letter_stats(logits, letter_ids: list[int], decode: Callable[[list[int]], str]) -> tuple[list[float], dict]:
    """From one next-token logits vector (torch): the letters' logits, the top-1 token over the whole
    vocabulary, and the probability mass the letters hold (how much the model 'wants' to answer)."""
    logits = logits.float()
    top = int(logits.argmax())
    mass = float(logits.softmax(-1)[letter_ids].sum())
    return [float(logits[i]) for i in letter_ids], {"top_token": decode([top]), "letter_mass": mass}


def text_prompt(parts: list[Part]) -> str:
    if len(parts) != 1 or not isinstance(parts[0], TextPart):
        raise TypeError(
            "local_hf generate takes one text prompt (e.g. text_verbatim); for option probabilities use mode: logprob"
        )
    return parts[0].text


class LocalHFClient:
    provider = "local_hf"
    request_settings: frozenset[str] = frozenset()  # thinking is always off; there is no reasoning to log

    def __init__(self, model: str, params: dict | None = None):
        self.model = model
        self.repo, self.revision, self.mode, self.template = check_params(params)
        self.backend = check_backend(params)
        self.lock = threading.Lock()
        if self.backend == "modal":
            self.remote = remote_model("LocalHF", repo=self.repo, revision=self.revision, mode=self.mode, template=self.template)
            return
        self.load()
        if self.mode == "logprob":
            letter_token_ids(self.encode, self.prompt("SCENARIO: check", LETTER_CHECK), option_letters(LETTER_CHECK))

    @property
    def served(self) -> str:
        return f"{self.repo}@{self.revision}"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device(torch)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        lm: Any = AutoModelForCausalLM.from_pretrained(self.repo, revision=self.revision, dtype=torch.bfloat16)
        self.lm = lm.to(self.device).eval()
        self.load_s = time.perf_counter() - t0
        print(f"  loaded {self.served} as {type(self.lm).__name__} (bf16, {self.device}) in {self.load_s:.1f} s")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def chat(self, content: str) -> str:
        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def prompt(self, state: str, question: DecisionQuestion) -> str:
        if self.template == "plain":
            return plain_prompt(state, question)
        return self.chat(question_text(state, question))

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse:
        if self.mode == "logprob":
            return generate_decision(self, parts)
        if self.backend == "modal":
            return remote_complete(self.remote, text_prompt(parts))
        return self.complete(text_prompt(parts))

    def decide(self, part: DecisionPart) -> Decided:
        if self.backend == "modal":
            return remote_decide(self.remote, part)
        by_value, chosen, questions = {}, {}, {}
        for q in part.questions:
            prompt = self.prompt(part.state, q)
            letter_ids = letter_token_ids(self.encode, prompt, option_letters(q))
            ids = self.encode(prompt)
            with self.lock:
                scores, info = self.letter_logits(ids, letter_ids)
            by_value[q.id] = option_probs(q, scores)
            chosen[q.id] = argmax(by_value[q.id])
            questions[q.id] = {"letter_logits": scores, **info, "input_tokens": len(ids)}
        n_in = sum(q["input_tokens"] for q in questions.values())
        raw = {"model": self.served, "template": self.template, "questions": questions}
        return Decided(by_value, raw, {"input_tokens": n_in, "output_tokens": 0}, self.served, chosen)

    def letter_logits(self, ids: list[int], letter_ids: list[int]) -> tuple[list[float], dict]:
        import torch

        with torch.inference_mode():
            out = self.lm(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=1)
        return letter_stats(out.logits[0, -1].cpu(), letter_ids, self.tokenizer.decode)

    def complete(self, text: str) -> ModelResponse:
        import torch

        ids = self.encode(self.chat(text))
        with self.lock, Timer() as t, torch.inference_mode():
            out = self.lm.generate(
                torch.tensor([ids], device=self.device),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=self.device),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy: deterministic, so the cached reply is the reply (WORKLOG 2026-10-13)
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new = out[0, len(ids) :].tolist()
        eos = self.lm.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        reason = generation_finish(len(new), MAX_NEW_TOKENS, bool(new) and new[-1] in eos)
        raw = {"model": self.served, "template": "chat", "thinking": "off", "decoding": "greedy",
               "max_new_tokens": MAX_NEW_TOKENS, "finish_reason": reason}  # fmt: skip
        return ModelResponse(
            text=self.tokenizer.decode(new, skip_special_tokens=True),
            model=self.served,
            usage={"input_tokens": len(ids), "output_tokens": len(new)},
            latency_s=t.seconds,
            raw=raw,
        )
