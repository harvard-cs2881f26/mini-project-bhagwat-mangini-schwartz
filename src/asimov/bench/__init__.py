"""Benchmarks, named by input type: text (Fig. 4), video (Fig. 5), image (Fig. 6).

Each benchmark module defines one `BENCHMARK = Benchmark(...)` from its own loader, request
builder and scorer. The runner only sees this interface, so adding a benchmark never touches
the runner or the model clients. `get_benchmark("video")` imports modules lazily, like
`asimov.models.get_client`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from asimov.bench.parts import Part

# One entry per data folder: a benchmark's `name` is its folder under data/ (tests rely on this).
# The two Image variants will be two entries, "image" and "image_rationale", built by one
# function, each with its own prompt_version so their saved replies never mix.
MODULES = {
    "text": "asimov.bench.text", "video": "asimov.bench.video",
    "image": "asimov.bench.image", "image_rationale": "asimov.bench.image",
}


@dataclass(frozen=True)
class Benchmark:
    name: str
    load_items: Callable[[], list]  # inputs only; each item has an `.id`
    load_answers: Callable[[], dict[str, Any]]  # answer key by id, for the scorer only
    build_request: Callable[[Any, str], list[Part]]  # (item, prompt name) -> request parts
    score: Callable[[dict[str, Any], dict[str, str]], dict]  # (answers, replies by id) -> {"n", metrics..., "per_item"}
    # prompt name -> text identifying the request in the saved-reply key. It must change whenever
    # anything our code sends changes, other than the item itself: prompt file text, prompt name or
    # variant, wrapper text, dataset version. Otherwise a new prompt silently reuses old replies.
    # Video: the prompt file's full text. Text: "asimov_v2_injuries/0.1.0|<prompt name>" (verbatim, no wrapper).
    prompt_version: Callable[[str], str]
    columns: tuple[tuple[str, str, bool], ...]  # (label, summary key, is percent) for scripts/run.py


def get_benchmark(name: str) -> Benchmark:
    if name not in MODULES:
        raise ValueError(f"unknown benchmark {name!r}; choose from {sorted(MODULES)}")
    module = import_module(MODULES[name])
    return module.BENCHMARKS[name] if hasattr(module, "BENCHMARKS") else module.BENCHMARK
