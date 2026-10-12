"""Shared reading of data/<benchmark>/items.jsonl, written by scripts/download_data.py."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# ASIMOV_DATA_DIR points the loaders elsewhere, e.g. at the 5-item test sample in tests/fixtures/data.
DATA_DIR = Path(os.environ.get("ASIMOV_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")


def read_rows(folder: str, data_dir: Path = DATA_DIR) -> list[dict]:
    """folder: 'text', 'video', 'image' or 'image_rationale'."""
    with open(data_dir / folder / "items.jsonl") as f:
        return [json.loads(line) for line in f]


def read_answer_json(raw: str) -> dict:
    """The answer key's fenced JSON block ('```json{...}```') -> dict."""
    m = re.search(r"\{.*\}", raw, re.S)
    if m is None:
        raise ValueError(f"no JSON object in answer: {raw!r}")
    return json.loads(m.group(0))
