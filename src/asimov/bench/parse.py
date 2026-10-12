"""Shared parsing of model replies (scoring conventions: CLAUDE.md)."""
from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Hashable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


LOOSE_PAIR = re.compile(r"""(["'])([\w -]+?)\1\s*:\s*(["'])([^\n]*?)\3\s*(?=[,}\n]|$)""")


def json_object(text: str) -> dict | None:
    """The answer object in a reply; tolerates code fences, prose around it and single quotes.
    A key repeated in an object keeps its first value; `duplicate_keys` says which were repeated.

    Tries, in order, and returns the first that loads as a dict:
    1. the last top-level {...}, nested objects included (so a reply with an extra sub-object
       still yields its main object, not the sub-object);
    2. the last {...} with nothing nested inside (catches the paper format's stray '}').
    None if the reply has no object that loads (broken JSON: see `loose_pairs`).
    """
    for candidate in [*reversed(top_level_objects(text)), *reversed(re.findall(r"\{[^{}]*\}", text, re.S))]:
        d = load_dict(candidate)
        if d is not None:
            return d
    return None


def load_dict(candidate: str) -> dict | None:
    for loader in (load_json, load_literal):
        try:
            d = loader(candidate)
        except (ValueError, SyntaxError, TypeError, RecursionError):  # e.g. {[1]: "x"}, absurd nesting
            continue
        if isinstance(d, dict):
            return d
    return None


class ReplyObject(dict):
    """A reply's JSON object, read with the first value kept for a repeated key (the paper's format
    puts each answer before its rationale). `duplicate_keys`: this object's own repeated keys."""

    duplicate_keys: frozenset[str] = frozenset()


def first_values(pairs: list[tuple[object, object]]) -> ReplyObject:
    """Build one object from its (key, value) pairs in order; a repeated key keeps its first value."""
    obj, repeated = ReplyObject(), set()
    for key, value in pairs:
        if key in obj:
            repeated.add(key)
        else:
            obj[key] = value
    obj.duplicate_keys = frozenset(repeated)
    return obj


def duplicate_keys(obj: dict | None) -> frozenset[str]:
    """Keys repeated in a reply's object (top level only); empty for a plain dict or None."""
    return getattr(obj, "duplicate_keys", frozenset())


def load_json(candidate: str) -> object:
    return json.loads(candidate, object_pairs_hook=first_values)


def load_literal(candidate: str) -> object:
    """`ast.literal_eval`, except that dict literals (at any depth) keep a repeated key's first value."""
    return literal(ast.parse(candidate.lstrip(" \t"), mode="eval").body)


def literal(node: ast.expr) -> object:
    if isinstance(node, ast.Dict):
        return first_values([(dict_key(k), literal(v)) for k, v in zip(node.keys, node.values, strict=True)])
    if isinstance(node, ast.List):
        return [literal(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(literal(e) for e in node.elts)
    return ast.literal_eval(node)


def dict_key(node: ast.expr | None) -> object:
    if node is None:  # {**x}: not a literal
        raise ValueError("malformed node or string: dict unpacking")
    return ast.literal_eval(node)


def top_level_objects(text: str) -> list[str]:
    """Each outermost balanced {...} span, in order. Braces inside strings aren't special-cased:
    a span that doesn't load is skipped by `json_object`, which then tries the flat objects."""
    spans, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return spans


def loose_pairs(text: str) -> dict | None:
    """`"key": "value"` pairs read straight from the text, for replies whose JSON won't load
    (e.g. an unescaped quote inside a rationale). First value per key wins; None if there are none.
    Scorers count answers read this way separately (e.g. a 'loose_json' status)."""
    pairs: dict[str, str] = {}
    for m in LOOSE_PAIR.finditer(text):
        pairs.setdefault(m.group(2), m.group(4))
    return pairs or None


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
