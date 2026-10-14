"""Leakage test for every registered benchmark: a request may only carry the prompt and the item.

Searching requests for answer values doesn't work: "Yes"/"No", letters A-D and field names are
legitimately in the prompts. Instead, for every item of every benchmark (with each prompt a config
uses), check that:
  1. the item's raw answer string never appears in the request (text or image bytes);
  2. the item's ID never appears in the request;
  3. no field of the item contains the raw answer (a loader must not put the answer on the item).
Stricter benchmark-specific checks live next to each benchmark (e.g. tests/test_video_request.py).
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from asimov.bench import MODULES, get_benchmark
from asimov.bench.parts import DecisionPart, ImagePart, TextPart
from asimov.data.rows import read_rows

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
RUN_CONFIGS = [c for c in (yaml.safe_load(open(p)) for p in CONFIGS) if "run" in c]  # not e.g. configs/prices.yaml
PROMPTS = sorted({(c["benchmark"], c["prompt"]) for c in RUN_CONFIGS})


def test_every_registered_benchmark_has_a_config():
    assert set(MODULES) <= {name for name, _ in PROMPTS}


@pytest.mark.parametrize("name,prompt", PROMPTS)
def test_no_answer_or_id_in_any_request(name, prompt):
    bench = get_benchmark(name)
    raw_answers = {r["id"]: [v for k, v in r.items() if k.endswith("_raw")] for r in read_rows(name)}
    items = bench.load_items()
    assert {i.id for i in items} == set(raw_answers)
    for item in items:
        raw_values = raw_answers[item.id]
        parts = bench.build_request(item, prompt)
        text = "\n".join(
            [p.text for p in parts if isinstance(p, TextPart)] + [decision_text(p) for p in parts if isinstance(p, DecisionPart)]
        )
        images = b"".join(p.data for p in parts if isinstance(p, ImagePart))
        assert raw_values
        assert all(raw not in text and raw.encode() not in images for raw in raw_values), f"answer leaks into {item.id}"
        assert item.id not in text, f"ID leaks into {item.id}"
        fields = {f.name: getattr(item, f.name) for f in dataclasses.fields(item) if f.name != "id"}
        assert not any(raw in str(v) for raw in raw_values for v in fields.values()), f"answer stored on item {item.id}"


def decision_text(part: DecisionPart) -> str:
    """Everything a decision request sends: state, instructions, option values and descriptions."""
    return "\n".join(
        [part.state] + [f"{q.id}\n{q.instructions}\n" + "\n".join(f"{v} {d}" for v, d in q.options) for q in part.questions]
    )
