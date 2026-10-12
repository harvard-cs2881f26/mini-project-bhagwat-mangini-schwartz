"""Shared parsing of model replies."""
from __future__ import annotations

import ast
import json
import re


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
