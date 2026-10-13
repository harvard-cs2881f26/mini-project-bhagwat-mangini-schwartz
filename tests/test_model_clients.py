"""Offline tests: each client converts request parts into its provider's format correctly.
No network and no keys needed."""
import base64

import pytest

from asimov.bench.parts import ImagePart, TextPart
from asimov.models import get_client
from asimov.models.anthropic_client import to_content
from asimov.models.gemini_client import min_thinking, to_contents
from asimov.models.openai_client import to_input

PARTS = [TextPart("intro"), TextPart("t = 0.0 s"), ImagePart(b"\xff\xd8jpeg0"), TextPart("questions")]


def test_anthropic_format_keeps_order_and_bytes():
    c = to_content(PARTS)
    assert [b["type"] for b in c] == ["text", "text", "image", "text"]
    assert base64.standard_b64decode(c[2]["source"]["data"]) == b"\xff\xd8jpeg0"
    assert c[2]["source"]["media_type"] == "image/jpeg"


def test_openai_format_full_resolution():
    msg = to_input(PARTS)[0]
    assert msg["role"] == "user"
    img = msg["content"][2]
    assert img["type"] == "input_image" and img["detail"] == "high"
    assert img["image_url"].startswith("data:image/jpeg;base64,")
    assert base64.standard_b64decode(img["image_url"].split(",", 1)[1]) == b"\xff\xd8jpeg0"


def test_gemini_format_and_min_thinking():
    c = to_contents(PARTS)
    assert c[0].text == "intro" and c[2].inline_data.data == b"\xff\xd8jpeg0"
    assert min_thinking("gemini-2.5-pro").thinking_budget == 128
    assert min_thinking("gemini-2.5-flash-lite").thinking_budget == 0
    assert min_thinking("gemini-3.1-pro-preview").thinking_level == "LOW"
    assert min_thinking("gemini-3.1-flash-lite").thinking_level == "MINIMAL"


def test_mock_clients():
    assert '"No"' in get_client("mock", "mock-always-no").generate(PARTS).text
    with pytest.raises(ValueError):
        get_client("mock", "mock-perfect")


def test_openai_min_effort_is_the_lowest_each_model_lists():
    from asimov.models.openai_client import min_effort

    assert min_effort("gpt-5-nano") == "minimal"
    assert min_effort("gpt-6-luna") == "none"
