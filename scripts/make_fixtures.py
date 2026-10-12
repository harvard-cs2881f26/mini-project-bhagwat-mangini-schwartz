#!/usr/bin/env python
"""Copy a 5-item sample of each benchmark from data/ into tests/fixtures/data/, for end-to-end
pipeline tests that run without the full download (e.g. on every PR).

    python scripts/make_fixtures.py

Same layout and schema as data/, so loaders read it unchanged (ASIMOV_DATA_DIR=tests/fixtures/data).
Items are the first 5 of the runner's fixed order (what --limit 5 runs); image and image_rationale
use the same 5 items (same number, e.g. image_0054 and image_rationale_0054). Video frames are
downscaled to keep the repo small: tests check which frames are sent, not their pixels. Real runs always use the original frames in data/.
CLAUDE.md allows at most 5 items per benchmark here (WORKLOG 2026-10-12).
"""
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.data.rows import read_rows  # noqa: E402
from asimov.runner import fixed_order  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "tests" / "fixtures" / "data"
N = 5  # hard cap, see CLAUDE.md
FRAME_PX = 96  # longest side of a downscaled video frame


class _Id:
    def __init__(self, id: str):
        self.id = id


def pick(rows: list[dict]) -> list[dict]:
    ids = {i.id for i in fixed_order([_Id(r["id"]) for r in rows])[:N]}
    return [r for r in rows if r["id"] in ids]


def small_jpeg(src: Path) -> bytes:
    im = Image.open(src)
    im.thumbnail((FRAME_PX, FRAME_PX))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=60)
    return buf.getvalue()


def write(folder: str, rows: list[dict]) -> None:
    out = OUT / folder
    with open(out / "items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for folder in ("text", "video", "image", "image_rationale"):
        (OUT / folder).mkdir(parents=True)

    write("text", pick(read_rows("text")))

    video = pick(read_rows("video"))
    for r in video:
        for i in range(r["n_frames"]):
            dest = OUT / "video" / r["frames_dir"] / f"{i:05d}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(small_jpeg(DATA / "video" / r["frames_dir"] / f"{i:05d}.jpg"))
    write("video", video)

    nums = {r["id"].rsplit("_", 1)[1] for r in pick(read_rows("image"))}
    for folder in ("image", "image_rationale"):
        rows = [r for r in read_rows(folder) if r["id"].rsplit("_", 1)[1] in nums]
        for r in rows:
            dest = OUT / folder / r["image"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DATA / folder / r["image"], dest)
        write(folder, rows)

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {N} items per benchmark to {OUT.relative_to(ROOT)} ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
