"""Benchmarks, named by input type: text (Fig. 4), video (Fig. 5), image (Fig. 6).

Each benchmark module defines one `BENCHMARK = Benchmark(...)` from its own loader, request
builder and scorer. The runner only sees this interface, so adding a benchmark never touches
the runner or the model clients. `get_benchmark("video")` imports modules lazily, like
`asimov.models.get_client`.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from asimov.bench.parts import Part

MODULES = {"video": "asimov.bench.video"}


@dataclass(frozen=True)
class Benchmark:
    name: str
    load_items: Callable[[], list]  # inputs only; each item has an `.id`
    load_answers: Callable[[], dict[str, Any]]  # answer key by id, for the scorer only
    build_request: Callable[[Any, str], list[Part]]  # (item, prompt name) -> request parts
    score: Callable[[dict[str, Any], dict[str, str]], dict]  # (answers, replies by id) -> {"n", metrics..., "per_item"}
    prompt_version: Callable[[str], str]  # prompt name -> text that identifies it in the saved-reply key
    columns: tuple[tuple[str, str, bool], ...]  # (label, summary key, is percent) for scripts/run.py


def get_benchmark(name: str) -> Benchmark:
    if name not in MODULES:
        raise ValueError(f"unknown benchmark {name!r}; choose from {sorted(MODULES)}")
    return import_module(MODULES[name]).BENCHMARK
