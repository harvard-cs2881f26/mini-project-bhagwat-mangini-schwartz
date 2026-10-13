"""Plain text for local decision models (Part 2, Step C), and the pure maths on their scores.

A local model takes text, not a DecisionPart, so each question is rendered with the `local`
section of prompts/decision_questions.yaml (the only home for that wording; bench.text.wording):
    question_text   state + question + options as "A. <description>" lines (+ the answer
                    instruction): the chat model's user turn
    plain_prompt    question_text + the answer cue: the base model's prompt, ending where the
                    answer letter goes
    hypotheses      NLI: one hypothesis per option (the premise is the state)
The letters are labels for this prompt only, in the order the options are sent; scores come back
in that order and are mapped to the option values sent (models.decision then maps values to
answer-key letters). No torch here, so all of it is tested offline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from string import ascii_uppercase

from asimov.bench.parts import DecisionQuestion
from asimov.bench.text import wording


def local_wording() -> dict:
    return wording()["local"]


def option_letters(question: DecisionQuestion) -> tuple[str, ...]:
    return tuple(ascii_uppercase[: len(question.options)])


def question_text(state: str, question: DecisionQuestion) -> str:
    w = local_wording()
    lines = [
        w["option"].format(letter=letter, description=d)
        for letter, (_, d) in zip(option_letters(question), question.options, strict=True)
    ]
    return w["question"].format(state=state, instructions=question.instructions, options="\n".join(lines))


def plain_prompt(state: str, question: DecisionQuestion) -> str:
    return question_text(state, question) + local_wording()["answer_cue"]


def hypotheses(question: DecisionQuestion) -> list[str]:
    template = local_wording()["nli_hypothesis"][question.id]
    return [template.format(description=d) for _, d in question.options]


def answer_token(prompt: str, letter: str) -> str:
    """The letter as the model would write it next: " A" after a word or colon, "A" after whitespace."""
    return letter if prompt[-1:].isspace() else " " + letter


def letter_token_ids(encode: Callable[[str], list[int]], prompt: str, letters: Sequence[str]) -> list[int]:
    """The token id of each answer letter written right after `prompt`. Fails loudly unless the
    letter is exactly one new token that doesn't merge with the prompt's end, and the ids differ."""
    base = encode(prompt)
    ids = []
    for letter in letters:
        full = encode(prompt + answer_token(prompt, letter))
        if len(full) != len(base) + 1 or full[: len(base)] != base:
            raise ValueError(
                f"answer letter {answer_token(prompt, letter)!r} after {prompt[-20:]!r} is not one token for this "
                "tokenizer; change the `local` prompt ending in prompts/decision_questions.yaml"
            )
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise ValueError(f"answer letters {list(letters)} share token ids {ids}")
    return ids


def softmax(scores: Sequence[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    return [e / sum(exps) for e in exps]


def option_probs(question: DecisionQuestion, scores: Sequence[float]) -> dict[str, float]:
    """Scores in the order the options were sent -> {option value: probability}, softmax over the options only."""
    if len(scores) != len(question.options):
        raise ValueError(f"{question.id}: {len(scores)} scores for {len(question.options)} options")
    return {v: p for (v, _), p in zip(question.options, softmax(scores), strict=True)}


def entailment_index(label2id: dict[str, int]) -> int:
    """The NLI model's entailment output, found by name like the zero-shot pipeline (first label starting 'entail')."""
    for label, index in label2id.items():
        if label.lower().startswith("entail"):
            return index
    raise ValueError(f"no entailment label in {label2id}; this is not an NLI checkpoint")
