#!/usr/bin/env python
"""Download the four ASIMOV-2.0 TFDS builds from GCS and convert them to plain files.

    python scripts/download_data.py            # all datasets
    python scripts/download_data.py text       # one dataset
    python scripts/download_data.py --force    # reconvert even if outputs exist

Benchmarks are named by input type: text (paper: Injury, Fig. 4), video (Fig. 5), image
(paper: Constraints, Fig. 6). DATASETS below is the only place the paper's dataset names appear.

Output (under data/):
    raw/<gcs_name>/...                      TFRecord + TFDS metadata, as downloaded
    text/items.jsonl                        {id, prompt, answer_raw}
    video/items.jsonl                       {id, prompt, answer_raw, frames_dir, timestamps, n_frames}
    video/frames/<id>/<NNNNN>.jpg           frames, original JPEG bytes (no re-encoding)
    image/items.jsonl                       {id, prompt, image, width, height,
    image_rationale/items.jsonl              violating_raw, non_violating_raw}
    <benchmark>/images/<id>.jpg             original JPEG bytes

Box strings are kept raw here and parsed by src/asimov/data/image.py, so a parser
fix never needs a re-download. Idempotent: files already present with the right size are
not re-downloaded, and a dataset whose items.jsonl already has the expected count is skipped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.data.tfrecord import read_examples  # noqa: E402

DATA = ROOT / "data"
BASE_URL = "https://storage.googleapis.com/gresearch/robotics/{name}/0.1.0/{file}"

DATASETS = {
    "text": ("asimov_v2_injuries", 319),
    "video": ("asimov_v2_videos", 287),
    "image": ("asimov_v2_constraints_without_rationale", 164),
    "image_rationale": ("asimov_v2_constraints_with_rationale", 164),
}


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def fetch(gcs_name: str) -> Path:
    raw = DATA / "raw" / gcs_name
    raw.mkdir(parents=True, exist_ok=True)
    files = ["dataset_info.json", "features.json", f"{gcs_name}-val.tfrecord-00000-of-00001"]
    for fn in files:
        url = BASE_URL.format(name=gcs_name, file=fn)
        dest = raw / fn
        size = _remote_size(url)
        if dest.exists() and dest.stat().st_size == size:
            continue
        print(f"  downloading {fn} ({size / 1e6:.1f} MB)")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if tmp.stat().st_size != size:
            raise RuntimeError(f"size mismatch for {fn}")
        tmp.rename(dest)
    return raw / files[-1]


def _text(ex: dict, key: str) -> str:
    return ex[key][0].decode("utf-8")


def _jpeg_size(b: bytes) -> tuple[int, int]:
    from PIL import Image
    import io

    with Image.open(io.BytesIO(b)) as im:
        return im.size  # (width, height)


def convert(bench: str, tfrecord: Path, out: Path) -> int:
    rows = []
    for i, ex in enumerate(read_examples(tfrecord)):
        item_id = f"{bench}_{i:04d}"
        row = {"id": item_id, "prompt": _text(ex, "prompt")}
        if bench == "text":
            row["answer_raw"] = _text(ex, "answer")
        elif bench == "video":
            row["answer_raw"] = _text(ex, "answer")
            frames = ex["video/frames"]
            fdir = out / "frames" / item_id
            fdir.mkdir(parents=True, exist_ok=True)
            for j, fb in enumerate(frames):
                (fdir / f"{j:05d}.jpg").write_bytes(fb)
            row["frames_dir"] = str(fdir.relative_to(out))
            row["timestamps"] = [t.decode("utf-8") for t in ex["video/timestamps"]]
            row["n_frames"] = len(frames)
        else:
            img = ex["image"][0]
            (out / "images").mkdir(parents=True, exist_ok=True)
            path = out / "images" / f"{item_id}.jpg"
            path.write_bytes(img)
            row["image"] = str(path.relative_to(out))
            row["width"], row["height"] = _jpeg_size(img)
            row["violating_raw"] = _text(ex, "violating_objects_bounding_boxes")
            row["non_violating_raw"] = _text(ex, "non_violating_objects_bounding_boxes")
        rows.append(row)
    with open(out / "items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    # default=None: newer Pythons check a list default against choices and reject it
    ap.add_argument("datasets", nargs="*", choices=list(DATASETS), help="default: all")
    ap.add_argument("--force", action="store_true", help="reconvert even if items.jsonl exists")
    args = ap.parse_args()

    ok = True
    for bench in args.datasets or list(DATASETS):
        gcs_name, expected = DATASETS[bench]
        out = DATA / bench
        items = out / "items.jsonl"
        if items.exists() and not args.force:
            n = sum(1 for _ in open(items))
            if n == expected:
                print(f"[{bench}] already converted ({n} items), skipping")
                continue
        print(f"[{bench}] {gcs_name}")
        tfrecord = fetch(gcs_name)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        n = convert(bench, tfrecord, out)
        status = "OK" if n == expected else f"MISMATCH (expected {expected})"
        ok &= n == expected
        print(f"[{bench}] {n} items -> {items.relative_to(ROOT)}  {status}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
