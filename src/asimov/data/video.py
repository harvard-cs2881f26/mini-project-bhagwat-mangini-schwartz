"""Video benchmark loader.

Inputs and answers are loaded by two separate functions on purpose: `load_video_items` returns
only what a model may see (frames and their timestamps), and `load_video_answers` returns the
answer key for the scorer. The request builder only ever receives `VideoItem`s, so it has no way
to put an answer into a request (see tests/test_video_request.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from asimov.data.rows import DATA_DIR, read_answer_json, read_rows


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


def load_video_items(data_dir: Path = DATA_DIR) -> list[VideoItem]:
    root = data_dir / "video"
    items = []
    for r in read_rows("video", data_dir):
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
    d = read_answer_json(raw)
    return VideoAnswer(
        risk=d["risk_of_physical_injury"],
        last_intervention_s=float(d["last_intervention_timestamp_seconds"]),
        injury_type=d["most_significant_injury_type"],
        severity=d["injury_severity"],
    )


def load_video_answers(data_dir: Path = DATA_DIR) -> dict[str, VideoAnswer]:
    """Answer key, for the scorer only. Never pass this to the request builder."""
    return {r["id"]: parse_answer_raw(r["answer_raw"]) for r in read_rows("video", data_dir)}
