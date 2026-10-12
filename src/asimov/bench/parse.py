"""Shared parsing of model replies (scoring conventions: CLAUDE.md)."""
from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Hashable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def json_object(text: str) -> dict | None:
    """The last {...} object in the reply; tolerates code fences and single quotes."""
    candidates = re.findall(r"\{[^{}]*\}", text, re.S)
    for c in reversed(candidates):
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(c)
                if isinstance(d, dict):
                    return d
            except (ValueError, SyntaxError):
                continue
    return None


def normalize_key(key: str) -> str:
    """'Latent Severity-Response ' -> 'latent_severity_response'."""
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def locate_keys(obj: dict, exact: dict[T, str], near: Callable[[str], T | None]) -> dict[T, tuple[str, bool]]:
    """Which key in a reply's JSON object holds each target: target -> (key, exact?).

    Format slips in key names shouldn't cost an answer when the intended target is clear:
    - a target's exact key always wins;
    - otherwise a near-miss key is used if it is the only one `near` maps to that target
      (`near` is the benchmark's own rule, e.g. "the key contains 'severity'");
    - two near-miss keys for one target leave it unread: we never guess between them.
    Scorers should count near-miss reads (e.g. a 'near_key' status) so slips stay visible.
    """
    found = {target: (key, True) for target, key in exact.items() if key in obj}
    candidates: dict[T, list[str]] = {}
    for key in obj:
        target = None if key in exact.values() else near(key)
        if target is not None and target not in found:
            candidates.setdefault(target, []).append(key)
    return found | {target: (keys[0], False) for target, keys in candidates.items() if len(keys) == 1}
