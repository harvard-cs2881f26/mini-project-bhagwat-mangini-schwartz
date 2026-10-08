"""Minimal pure-Python TFRecord + tf.train.Example reader.

Only used by scripts/download_data.py. Nothing else in the repo touches TFRecords.

TFRecord framing per record:
    uint64 length | uint32 masked_crc32c(length) | bytes data[length] | uint32 masked_crc32c(data)
CRCs are not verified (crc32c is not in the stdlib); a truncated file raises instead.

tf.train.Example protobuf (only the parts we need):
    Example   { Features features = 1; }
    Features  { map<string, Feature> feature = 1; }   # map entry: key = 1, value = 2
    Feature   { oneof { BytesList bytes_list = 1; FloatList float_list = 2; Int64List int64_list = 3; } }
    BytesList { repeated bytes value = 1; }
TFDS flattens nested features with "/" (e.g. "video/frames").
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator


def iter_records(path: str | Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) < 12:
                raise ValueError(f"truncated record header in {path}")
            (length,) = struct.unpack("<Q", header[:8])
            data = f.read(length)
            if len(data) != length or len(f.read(4)) != 4:
                raise ValueError(f"truncated record body in {path}")
            yield data


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 1:
            val, i = buf[i : i + 8], i + 8
        elif wire == 2:
            ln, i = _varint(buf, i)
            val, i = buf[i : i + ln], i + ln
        elif wire == 5:
            val, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, val


def _parse_feature(buf: bytes) -> list:
    for field, _, val in _fields(buf):
        if field == 1:  # BytesList
            return [v for f, _, v in _fields(val) if f == 1]
        if field == 2:  # FloatList (packed or not)
            out = []
            for f, w, v in _fields(val):
                if w == 2:
                    out += list(struct.unpack(f"<{len(v) // 4}f", v))
                else:
                    out.append(struct.unpack("<f", v)[0])
            return out
        if field == 3:  # Int64List (packed or not)
            out = []
            for f, w, v in _fields(val):
                if w == 2:
                    j = 0
                    while j < len(v):
                        x, j = _varint(v, j)
                        out.append(x)
                else:
                    out.append(v)
            return out
    return []


def parse_example(buf: bytes) -> dict[str, list]:
    """Return {feature_name: list of values} for one serialized tf.train.Example."""
    out: dict[str, list] = {}
    for field, _, features in _fields(buf):
        if field != 1:
            continue
        for f, _, entry in _fields(features):
            if f != 1:
                continue
            key, value = None, b""
            for ef, _, ev in _fields(entry):
                if ef == 1:
                    key = ev.decode("utf-8")
                elif ef == 2:
                    value = ev
            out[key] = _parse_feature(value)
    return out


def read_examples(path: str | Path) -> Iterator[dict[str, list]]:
    for rec in iter_records(path):
        yield parse_example(rec)
