#!/usr/bin/env python
"""Overlay constraint boxes on images for a human check. Red = violating, green = safe,
grey dashed = dropped as degenerate.

    python scripts/draw_boxes.py                       # 5 evenly spaced items
    python scripts/draw_boxes.py image_0032 ...        # specific items
    python scripts/draw_boxes.py --n 20 --out results/box_check

Output: one PNG per item plus a text file listing each box's label, so the reviewer can
check that each label matches the object its box sits on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.data.image import Box, load_image_items  # noqa: E402

RED, GREEN, GREY = (230, 30, 30), (20, 200, 60), (160, 160, 160)


def _px(b: Box, w: int, h: int) -> tuple[float, float, float, float]:
    return (b.xmin / 1000 * w, b.ymin / 1000 * h, b.xmax / 1000 * w, b.ymax / 1000 * h)


def draw(item, out_dir: Path) -> Path:
    im = Image.open(item.image_path).convert("RGB")
    scale = max(1, round(1000 / im.width))  # upscale small images so labels are legible
    im = im.resize((im.width * scale, im.height * scale))
    d = ImageDraw.Draw(im)
    lines = [item.prompt.split("<CONSTRAINTS END>")[0], ""]
    for n, b in enumerate(item.violating + item.safe + item.dropped):
        color = GREY if b in item.dropped else RED if b.violating else GREEN
        box = _px(b, im.width, im.height)
        d.rectangle(box, outline=color, width=1 if color == GREY else 2)
        d.text((box[0] + 3, box[1] + 2), str(n), fill=color)
        kind = "DROPPED" if color == GREY else "VIOLATING" if b.violating else "safe"
        lines.append(f"[{n}] {kind:9s} {b.label}")
    png = out_dir / f"{item.id}.png"
    im.save(png)
    (out_dir / f"{item.id}.txt").write_text("\n".join(lines) + "\n")
    return png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "box_check"))
    args = ap.parse_args()
    items = load_image_items()
    if args.ids:
        items = [i for i in items if i.id in set(args.ids)]
    else:
        step = len(items) // args.n
        items = items[::step][: args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for it in items:
        print(draw(it, out))


if __name__ == "__main__":
    main()
