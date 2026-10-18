"""Image constraint requests and registration with the shared benchmark runner.

No answer boxes are included in requests. Parsing is local policy, not an authors'
released evaluator; see WORKLOG.md.
"""

from __future__ import annotations

import hashlib
import json
from functools import partial

from asimov.bench import Benchmark
from asimov.bench.image_score import score
from asimov.bench.parts import ImagePart, Part, TextPart
from asimov.data.image import ImageInput, ImageItem, load_image_answers, load_image_inputs


def build_request(item: ImageInput | ImageItem, prompt_name: str = "image") -> list[Part]:
    if item.prompt.count("<1>") != 1:
        raise ValueError(f"{item.id}: expected exactly one <1> image marker")
    before, after = item.prompt.split("<1>")
    return [TextPart(before), ImagePart(item.image_path.read_bytes()), TextPart(after)]


def build_system_request(item: ImageInput | ImageItem) -> tuple[str, list[Part]]:
    """Move the exact constraint prefix into system content; preserve all text."""
    parts = build_request(item)
    assert isinstance(parts[0], TextPart)
    before, marker, after = parts[0].text.partition("<CONSTRAINTS END>")
    if not marker or before.count("<CONSTRAINTS START>") != 1 or item.prompt.count(marker) != 1:
        raise ValueError(f"{item.id}: expected one constraint block before the image")
    return before + marker, [TextPart(after), *parts[1:]]


def request_key(
    item: ImageInput,
    provider: str,
    model: str,
    thinking: str,
    variant: str,
    message_roles: str,
    generation_settings: dict | None = None,
) -> str:
    """Exact image request fingerprint; retain keys of previously paid calls."""
    legacy_variant = {"image": "constraints", "image_rationale": "constraints_rationale"}[variant]
    h = hashlib.sha256()
    for value in ("constraints-request-v1", provider, model, thinking, legacy_variant, item.prompt):
        h.update(value.encode())
        h.update(b"\0")
    if message_roles != "user":
        h.update(f"message_roles:{message_roles}\0".encode())
    if generation_settings:
        h.update(json.dumps(generation_settings, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\0")
    h.update(item.image_path.read_bytes())
    return h.hexdigest()


def _benchmark(variant: str) -> Benchmark:
    return Benchmark(
        name=variant,
        load_items=partial(load_image_inputs, variant),
        load_answers=partial(load_image_answers, variant),
        build_request=build_request,
        score=score,
        prompt_version=lambda prompt: f"asimov_v2_constraints/0.1.0|{variant}|{prompt}",
        columns=(
            ("violations", "violation_rate", True),
            ("parsed", "n_parsed", False),
            ("empty", "n_empty", False),
            ("parse fail", "parse_failure_rate", True),
        ),
    )


BENCHMARKS = {v: _benchmark(v) for v in ("image", "image_rationale")}
BENCHMARK = BENCHMARKS["image"]
