"""Video benchmark loader.

Inputs and answers are loaded by two separate functions on purpose: `load_video_items` returns
only what a model may see (frames and their timestamps), and `load_video_answers` returns the
answer key for the scorer. The request builder only ever receives `VideoItem`s, so it has no way
to put an answer into a request (see tests/test_video_request.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class VideoItem:
    id: str
    frame_paths: tuple[Path, ...]
    frame_seconds: tuple[float, ...]  # 0.0, 0.5, ..., 7.5


@dataclass(frozen=True)
class VideoAnswer:
    risk: str  # "Yes" | "No"
    last_intervention_s: float  # -1 = no intervention expected
    injury_type: str  # "A".."E" | "No_Consensus"
    severity: str  # "Mild Harm" | "Moderate Harm" | "Severe Harm" | "No_Consensus"


def _seconds(ts: str) -> float:
    """'00:01.500' -> 1.5"""
    minutes, seconds = ts.split(":")
    return int(minutes) * 60 + float(seconds)


def _rows(data_dir: Path) -> list[dict]:
    with open(data_dir / "video" / "items.jsonl") as f:
        return [json.loads(line) for line in f]


def load_video_items(data_dir: Path = DATA_DIR) -> list[VideoItem]:
    root = data_dir / "video"
    items = []
    for r in _rows(data_dir):
        fdir = root / r["frames_dir"]
        items.append(
            VideoItem(
                id=r["id"],
                frame_paths=tuple(fdir / f"{i:05d}.jpg" for i in range(r["n_frames"])),
                frame_seconds=tuple(_seconds(t) for t in r["timestamps"]),
            )
        )
    return items


def parse_answer_raw(raw: str) -> VideoAnswer:
    m = re.search(r"\{.*\}", raw, re.S)
    if m is None:
        raise ValueError(f"no JSON object in answer: {raw!r}")
    d = json.loads(m.group(0))
    return VideoAnswer(
        risk=d["risk_of_physical_injury"],
        last_intervention_s=float(d["last_intervention_timestamp_seconds"]),
        injury_type=d["most_significant_injury_type"],
        severity=d["injury_severity"],
    )


def load_video_answers(data_dir: Path = DATA_DIR) -> dict[str, VideoAnswer]:
    """Answer key, for the scorer only. Never pass this to the request builder."""
    return {r["id"]: parse_answer_raw(r["answer_raw"]) for r in _rows(data_dir)}
