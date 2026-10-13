"""NLI zero-shot classification, locally (Part 2, Step C): e.g. MoritzLaurer/deberta-v3-large-zeroshot-v2.0.

The zero-shot pipeline's logic with multi_label=False (docs/apis/nli_zeroshot.md): per question,
premise = the state, one hypothesis per option (models.local_text.hypotheses); the entailment
logits of all the options, softmaxed together, are the option probabilities. A decision client:
question modes, ties and the 4-letter reply come from models.decision. Unlike the pipeline, an
input longer than the model's limit raises instead of being cut silently.

params (config): repo (Hugging Face repo id), revision (the commit sha to load), and optionally
backend: local (default, load here) | modal (the deployed Modal app runs this same client; models/remote.py).
torch and transformers are imported only when a model is loaded (requirements-local.txt).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from asimov.bench.parts import DecisionPart, Part
from asimov.models.base import ModelResponse, Thinking
from asimov.models.decision import Decided, argmax, generate_decision
from asimov.models.local_hf import check_backend, device
from asimov.models.local_text import entailment_index, hypotheses, option_probs
from asimov.models.remote import remote_decide, remote_model

PARAMS = ("repo", "revision")


def check_params(params: dict | None) -> tuple[str, str]:
    params = {k: v for k, v in (params or {}).items() if k != "backend"}  # backend: check_backend
    if set(params) != set(PARAMS) or not all(params.values()):
        raise ValueError(f"nli params: give exactly {PARAMS}, got {sorted(params)}")
    return str(params["repo"]), str(params["revision"])


class NLIClient:
    provider = "nli"
    request_settings: frozenset[str] = frozenset()  # no thinking, no reasoning: thinking and log_reasoning are ignored

    def __init__(self, model: str, params: dict | None = None):
        self.model = model
        self.repo, self.revision = check_params(params)
        self.backend = check_backend(params)
        self.lock = threading.Lock()
        if self.backend == "modal":
            self.remote = remote_model("NLI", repo=self.repo, revision=self.revision)
            return
        self.load()

    @property
    def served(self) -> str:
        return f"{self.repo}@{self.revision}"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device(torch)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        nli: Any = AutoModelForSequenceClassification.from_pretrained(self.repo, revision=self.revision)
        self.nli = nli.to(self.device).eval()
        self.entail = entailment_index(self.nli.config.label2id)
        self.load_s = time.perf_counter() - t0
        print(f"  loaded {self.served} (fp32, {self.device}, entailment = output {self.entail}) in {self.load_s:.1f} s")

    def generate(self, parts: list[Part], thinking: Thinking = "min", log_reasoning: bool = False) -> ModelResponse:
        return generate_decision(self, parts)

    def decide(self, part: DecisionPart) -> Decided:
        if self.backend == "modal":
            return remote_decide(self.remote, part)
        by_value, chosen, questions = {}, {}, {}
        for q in part.questions:
            with self.lock:
                logits, n_tokens = self.entailment_logits(part.state, hypotheses(q))
            by_value[q.id] = option_probs(q, logits)
            chosen[q.id] = argmax(by_value[q.id])
            questions[q.id] = {"entailment_logits": logits, "input_tokens": n_tokens}
        n_in = sum(q["input_tokens"] for q in questions.values())
        raw = {"model": self.served, "questions": questions}
        return Decided(by_value, raw, {"input_tokens": n_in, "output_tokens": 0}, self.served, chosen)

    def entailment_logits(self, premise: str, hyps: list[str]) -> tuple[list[float], int]:
        """One batch of (premise, hypothesis) pairs -> each pair's entailment logit, and the tokens read."""
        import torch

        enc = self.tokenizer([premise] * len(hyps), hyps, padding=True, return_tensors="pt")
        if enc["input_ids"].shape[1] > self.tokenizer.model_max_length:
            raise ValueError(f"premise + hypothesis is {enc['input_ids'].shape[1]} tokens, over this model's "
                             f"{self.tokenizer.model_max_length}; it would be cut silently")  # fmt: skip
        with torch.inference_mode():
            logits = self.nli(**enc.to(self.device)).logits
        return logits[:, self.entail].float().cpu().tolist(), int(enc["attention_mask"].sum())
