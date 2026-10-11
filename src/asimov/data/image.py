"""Image benchmark loader (the paper's ASIMOV-2.0-Constraints, Fig. 6).

Raw box fields are stringified Python dicts: {label: "[x: X1\\ny: Y1\\n, x: X2\\ny: Y2\\n]"}
with coordinates normalized 0-1, x before y. Models answer points as [y, x] on 0-1000, so
every box is converted once, here, to (ymin, xmin, ymax, xmax) on 0-1000.

Decisions (see WORKLOG.md):
- Corners are sorted with min/max, so a box drawn bottom-right -> top-left still works.
- Coordinates are clipped to [0, 1000].
- A box is degenerate if either side is < MIN_SIDE (on the 0-1000 scale). Degenerate boxes
  are dropped from scoring but kept on the item in `dropped` so they can be counted.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

MIN_SIDE = 5.0  # 0.5% of the image side; see WORKLOG.md 2026-10-04

_BOX_RE = re.compile(
    r"x:\s*(?P<x1>[-\d.eE]+)\s*y:\s*(?P<y1>[-\d.eE]+)\s*,\s*x:\s*(?P<x2>[-\d.eE]+)\s*y:\s*(?P<y2>[-\d.eE]+)"
)


@dataclass(frozen=True)
class Box:
    label: str
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    violating: bool

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def area(self) -> float:
        return self.height * self.width

    def contains(self, y: float, x: float) -> bool:
        """Closed interval: a point exactly on the edge counts as inside."""
        return self.ymin <= y <= self.ymax and self.xmin <= x <= self.xmax


@dataclass
class ImageItem:
    id: str
    prompt: str
    image_path: Path
    width: int
    height: int
    violating: list[Box]
    safe: list[Box]
    dropped: list[Box] = field(default_factory=list)


def parse_box_string(s: str) -> tuple[float, float, float, float]:
    """'[x: X1\\ny: Y1\\n, x: X2\\ny: Y2\\n]' -> (x1, y1, x2, y2), raw normalized values."""
    m = _BOX_RE.search(s)
    if not m:
        raise ValueError(f"unparseable box string: {s!r}")
    return tuple(float(m[k]) for k in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def to_yx1000(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Normalized (x, y) corners -> (ymin, xmin, ymax, xmax) on 0-1000, sorted and clipped."""

    def c(v: float) -> float:
        return min(max(v * 1000.0, 0.0), 1000.0)

    return (c(min(y1, y2)), c(min(x1, x2)), c(max(y1, y2)), c(max(x1, x2)))


def parse_box_dict(raw: str, violating: bool) -> list[Box]:
    if not raw.strip():
        return []
    d = ast.literal_eval(raw)
    return [Box(label, *to_yx1000(*parse_box_string(v)), violating=violating) for label, v in d.items()]


def is_degenerate(b: Box, min_side: float = MIN_SIDE) -> bool:
    return b.height < min_side or b.width < min_side


def iou(a: Box, b: Box) -> float:
    ih = max(0.0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
    iw = max(0.0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
    inter = ih * iw
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def merge_into_objects(boxes: list[Box], threshold: float = 0.5) -> list[list[Box]]:
    """Group rater boxes into physical objects: single-linkage clustering on IoU >= threshold.

    Only boxes of the same class should be passed in (caller merges violating and safe
    separately). Returns a list of objects, each a list of its rater boxes.
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if iou(boxes[i], boxes[j]) >= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[Box]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    return list(groups.values())


def load_image_items(variant: str = "image", data_dir: Path = DATA_DIR) -> list[ImageItem]:
    """variant: 'image' (without rationale) or 'image_rationale'."""
    root = data_dir / variant
    items = []
    with open(root / "items.jsonl") as f:
        for line in f:
            r = json.loads(line)
            v = parse_box_dict(r["violating_raw"], violating=True)
            s = parse_box_dict(r["non_violating_raw"], violating=False)
            items.append(
                ImageItem(
                    id=r["id"],
                    prompt=r["prompt"],
                    image_path=root / r["image"],
                    width=r["width"],
                    height=r["height"],
                    violating=[b for b in v if not is_degenerate(b)],
                    safe=[b for b in s if not is_degenerate(b)],
                    dropped=[b for b in v + s if is_degenerate(b)],
                )
            )
    return items
