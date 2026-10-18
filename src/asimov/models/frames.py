"""Frame preparation for decision services that can't take the video's 16 full-size frames as-is.

shrink: Jev rejects 16 frames at 1280x720 (`max_tokens_exceeded`); its images must be 280-1568 px a side,
    so they are resized (e.g. to 498x280) and re-encoded as JPEG.
tiles:  Cloudflare's Clef takes at most 4 images per request, so 4 consecutive frames go in each image
    (2x2, read left to right, top to bottom), each with its label drawn in its top-left corner.
Both only resize or rearrange the stored frames; nothing else about the request changes.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from asimov.bench.parts import Frame

TILE_FRAME_SIZE = (640, 360)  # a 2x2 tile is 1280x720, about 1 megapixel: Clef's per-image cap (1,024 tokens)


def jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def shrink(frame: Frame, size: tuple[int, int]) -> bytes:
    return jpeg(Image.open(io.BytesIO(frame.data)).convert("RGB").resize(size))


def tiles(frames: tuple[Frame, ...], per_tile: int = 4) -> list[bytes]:
    w, h = TILE_FRAME_SIZE
    out = []
    for start in range(0, len(frames), per_tile):
        tile = Image.new("RGB", (2 * w, 2 * h))
        for k, f in enumerate(frames[start : start + per_tile]):
            img = Image.open(io.BytesIO(f.data)).convert("RGB").resize((w, h))
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 150, 34), fill="black")
            draw.text((8, 6), f.label, fill="white", font_size=24)
            tile.paste(img, ((k % 2) * w, (k // 2) * h))
        out.append(jpeg(tile))
    return out


def tile_note(frames: tuple[Frame, ...], per_tile: int = 4) -> str:
    """The sentence telling the model how the frames are laid out in the tiles."""
    spans = [f"image {i + 1} holds {frames[s].label.removeprefix('t = ')} to {frames[min(s + per_tile, len(frames)) - 1].label.removeprefix('t = ')}"
             for i, s in enumerate(range(0, len(frames), per_tile))]
    return (f"The {len(frames)} frames come as {len(spans)} images of 2x2 frames each, read left to right, top to bottom: "
            + ", ".join(spans) + ". Each frame's timestamp is written in its top-left corner.")
