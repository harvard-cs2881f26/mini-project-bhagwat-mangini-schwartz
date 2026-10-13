"""scripts/check_paper_version.py: version parsing and comparison on recorded Atom replies. No network."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_paper_version import is_newer, latest_version  # noqa: E402  # pyright: ignore[reportMissingImports]


def atom(entry_id: str) -> str:
    """A trimmed arXiv API reply with one entry."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<id>https://arxiv.org/api/abc</id>"
        f"<entry><id>{entry_id}</id><title>Can AI Perceive Physical Danger and Intervene?</title></entry>"
        "</feed>"
    )


def test_same_version() -> None:
    latest = latest_version(atom("http://arxiv.org/abs/2509.21651v2"))
    assert latest == "v2"
    assert not is_newer(latest, "v2")


def test_newer_version() -> None:
    latest = latest_version(atom("http://arxiv.org/abs/2509.21651v3"))
    assert latest == "v3"
    assert is_newer(latest, "v2")


def test_versions_compare_as_numbers() -> None:
    assert is_newer("v10", "v9")
    assert not is_newer("v9", "v10")


def test_malformed_xml_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="not valid XML"):
        latest_version("<feed><entry>")


def test_feed_without_entry_raises() -> None:
    with pytest.raises(ValueError, match="no <entry><id>"):
        latest_version('<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
