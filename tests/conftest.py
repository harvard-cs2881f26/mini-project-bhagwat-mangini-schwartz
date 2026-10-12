"""Pick the data the tests read, before any asimov module is imported.

- ASIMOV_DATA_DIR set: use it.
- Otherwise the full download in data/ if every benchmark is there,
- else the committed 5-item sample in tests/fixtures/data (scripts/make_fixtures.py).

So pipeline tests always run, in a fresh checkout or on CI too. Tests marked `full_data` check
facts about the whole dataset (item counts) and are skipped, with that reason, on the sample.
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("text", "video", "image", "image_rationale")
FIXTURES = ROOT / "tests" / "fixtures" / "data"

if not os.environ.get("ASIMOV_DATA_DIR"):
    full = all((ROOT / "data" / f / "items.jsonl").exists() for f in FOLDERS)
    os.environ["ASIMOV_DATA_DIR"] = str(ROOT / "data" if full else FIXTURES)

ON_SAMPLE = Path(os.environ["ASIMOV_DATA_DIR"]).resolve() == FIXTURES.resolve()


def pytest_report_header(config):
    return f"data: {os.environ['ASIMOV_DATA_DIR']}" + (" (5-item sample)" if ON_SAMPLE else "")


def pytest_configure(config):
    config.addinivalue_line("markers", "full_data: needs the full dataset in data/, not the 5-item sample")


def pytest_collection_modifyitems(config, items):
    if ON_SAMPLE:
        skip = pytest.mark.skip(reason="needs the full dataset (running on the 5-item sample)")
        for item in items:
            if "full_data" in item.keywords:
                item.add_marker(skip)
