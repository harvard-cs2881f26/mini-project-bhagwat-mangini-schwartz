"""Provider-neutral request parts. Every benchmark's `build_request` returns a list of these,
and each provider client turns them into its own message format."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"


Part = TextPart | ImagePart
