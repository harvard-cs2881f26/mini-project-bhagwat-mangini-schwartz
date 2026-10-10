"""Video benchmark: build the model request for one video.

A request is a provider-neutral list of parts, in order:
    text   prompt up to {FRAMES}
    16 x  (text "t = 1.5 s", image JPEG bytes)
    text   prompt after {FRAMES}
Each provider client turns these parts into its own message format. Images are sent unchanged
(full resolution, original JPEG bytes) and unnamed: no video ID or file name is ever sent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from asimov.data.video import VideoItem

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_PROMPT = "video_v1"


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"


Part = TextPart | ImagePart


def load_prompt(name: str = DEFAULT_PROMPT) -> str:
    template = (PROMPTS_DIR / f"{name}.txt").read_text()
    if template.count("{FRAMES}") != 1:
        raise ValueError(f"prompt {name} must contain {{FRAMES}} exactly once")
    return template


def frame_label(seconds: float) -> str:
    return f"t = {seconds:.1f} s"


def build_request(item: VideoItem, prompt_name: str = DEFAULT_PROMPT) -> list[Part]:
    before, after = load_prompt(prompt_name).split("{FRAMES}")
    parts: list[Part] = [TextPart(before.rstrip("\n"))]
    for seconds, path in zip(item.frame_seconds, item.frame_paths):
        parts.append(TextPart(frame_label(seconds)))
        parts.append(ImagePart(path.read_bytes()))
    parts.append(TextPart(after.lstrip("\n")))
    return parts
