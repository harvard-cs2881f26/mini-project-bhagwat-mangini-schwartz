"""Leakage test (H): a model must only ever see the prompt and that video's frames.

Searching requests for answer words doesn't work: "Yes", the JSON field names and frame labels
like "t = 4.0 s" are legitimately in every request. Instead we check that nothing per-video
except the frames can be in a request:
  1. the text of every request is identical across all videos;
  2. the images are exactly that video's frame files, in order;
  3. the raw answer string never appears anywhere in a request.
"""
import pytest

from asimov.bench.parts import ImagePart, TextPart
from asimov.bench.video import build_request, frame_label
from asimov.data.video import DATA_DIR, VideoItem, load_video_answers, load_video_items, parse_answer_raw



def _texts(parts):
    return [p.text for p in parts if isinstance(p, TextPart)]


def _images(parts):
    return [p.data for p in parts if isinstance(p, ImagePart)]


def test_request_structure(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i:05d}.jpg"
        p.write_bytes(f"img{i}".encode())
        paths.append(p)
    item = VideoItem("video_9999", tuple(paths), (0.0, 0.5, 1.0))
    parts = build_request(item)
    assert isinstance(parts[0], TextPart) and isinstance(parts[-1], TextPart)
    middle = parts[1:-1]
    assert [type(p) for p in middle] == [TextPart, ImagePart] * 3
    assert [p.text for p in middle[::2]] == ["t = 0.0 s", "t = 0.5 s", "t = 1.0 s"]
    assert [p.data for p in middle[1::2]] == [b"img0", b"img1", b"img2"]
    assert "{FRAMES}" not in "".join(_texts(parts))
    assert "video_9999" not in "".join(_texts(parts))


def test_frame_label():
    assert frame_label(0) == "t = 0.0 s" and frame_label(7.5) == "t = 7.5 s"


def test_parse_answer_raw_real_format():
    raw = '```json{"risk_of_physical_injury": "Yes",\n "last_intervention_timestamp_seconds": 4.0,\n "most_significant_injury_type": "A",\n "injury_severity": "No_Consensus"}```'
    a = parse_answer_raw(raw)
    assert (a.risk, a.last_intervention_s, a.injury_type, a.severity) == ("Yes", 4.0, "A", "No_Consensus")


def test_no_answer_leaks_into_any_request():
    items = load_video_items()
    assert items
    raw_answers = {}
    import json
    with open(DATA_DIR / "video" / "items.jsonl") as f:
        for line in f:
            r = json.loads(line)
            raw_answers[r["id"]] = r["answer_raw"]

    reference_text = None
    for item in items:
        parts = build_request(item)
        text = _texts(parts)
        # 1. text identical for every video
        if reference_text is None:
            reference_text = text
        assert text == reference_text, f"request text differs for {item.id}"
        # 2. images are exactly this video's frames, in order
        assert _images(parts) == [p.read_bytes() for p in item.frame_paths]
        assert len(_images(parts)) == 16
        # 3. the raw answer string never appears, and neither does the video ID
        blob = "\n".join(text)
        assert raw_answers[item.id] not in blob
        assert item.id not in blob
        assert raw_answers[item.id].encode() not in b"".join(_images(parts))


def test_answer_key_loads():
    answers = load_video_answers()
    assert answers.keys() == {i.id for i in load_video_items()}
    assert {a.risk for a in answers.values()} <= {"Yes", "No"}
