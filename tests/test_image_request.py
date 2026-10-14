import json
from dataclasses import replace

import pytest

from asimov.bench import get_benchmark
from asimov.bench.image import build_request
from asimov.bench.image_score import parse_reply
from asimov.bench.parts import ImagePart, TextPart
from asimov.data.image import ImageItem
from asimov.runner import run


def patch_image(monkeypatch, items):
    monkeypatch.setattr(
        "asimov.runner.get_benchmark",
        lambda variant: replace(get_benchmark(variant), load_items=lambda: items, load_answers=lambda: {i.id: i for i in items}),
    )


def item(tmp_path):
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"original jpeg bytes")
    return ImageItem("test_0", "Rules. Image <1>. Output [y, x].", image, 10, 10, [], [])


def test_request_preserves_prompt_and_image(tmp_path):
    i = item(tmp_path)
    parts = build_request(i)
    assert [type(p) for p in parts] == [TextPart, ImagePart, TextPart]
    assert isinstance(parts[0], TextPart) and isinstance(parts[2], TextPart)
    assert isinstance(parts[1], ImagePart)
    assert parts[0].text + "<1>" + parts[2].text == i.prompt
    assert parts[1].data == i.image_path.read_bytes()
    i.prompt += "<1>"
    with pytest.raises(ValueError):
        build_request(i)


def test_parser_wrappers_empty_and_axes():
    pred = parse_reply('Reasoning. ```json\n[{"point": [0, 1000], "label": "[block]"}]\n```')
    assert pred.error is None
    assert pred.points is not None
    assert (pred.points[0].y, pred.points[0].x) == (0, 1000)
    assert parse_reply("[]").points == ()
    assert parse_reply("I cannot help").points is None


@pytest.mark.parametrize("coords", [[True, 2], [float("nan"), 1], [float("inf"), 1], [-1, 2], [2, 1001], ["2", 3], [1]])
def test_invalid_coordinates_are_not_repaired(coords):
    assert parse_reply(json.dumps([{"point": coords, "label": "x"}])).points is None


def test_parser_does_not_salvage_partial_arrays():
    good = '{"point": [1, 2], "label": "x"}'
    for text in (f"[{good}, broken]", f"[{good},", f"[{good}, {{}}]", f"[{good}] then [broken]"):
        assert parse_reply(text).points is None


def test_constraints_cache_and_input_changes(tmp_path, monkeypatch):
    i = item(tmp_path)
    patch_image(monkeypatch, [i])
    cfg = {
        "run": "t",
        "benchmark": "image",
        "thinking": "min",
        "prompt": "image",
        "models": [
            {"name": "Points", "provider": "mock", "model": "mock-points"},
            {"name": "Broken", "provider": "mock", "model": "mock-error"},
        ],
    }
    rows = run(cfg, results_dir=tmp_path)
    assert rows[0]["n"] == 1 and rows[0]["parse_failure_rate"] == 0
    assert rows[0]["scoring_status"] == "literal_any_violating_box_interior"
    assert rows[1]["n"] == 0 and rows[1]["errors"] == 1
    path = tmp_path / "t/mock-points/responses.jsonl"
    original = path.read_text()
    run(cfg, results_dir=tmp_path)
    assert path.read_text() == original
    i.prompt += " changed"
    run(cfg, results_dir=tmp_path)
    assert len(path.read_text().splitlines()) == 2
    i.image_path.write_bytes(b"changed image")
    run(cfg, results_dir=tmp_path)
    assert len(path.read_text().splitlines()) == 3
    cfg["variant"] = "image_rationale"
    cfg["run"] = "rationale"
    run(cfg, results_dir=tmp_path)
    assert len((tmp_path / "rationale/mock-points/responses.jsonl").read_text().splitlines()) == 1


def test_unknown_benchmark_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown benchmark"):
        run({"run": "bad", "benchmark": "typo", "prompt": "image", "thinking": "min", "models": []}, results_dir=tmp_path)


def test_renamed_item_reuses_legacy_cache(tmp_path, monkeypatch):
    from asimov import runner

    i = item(tmp_path)
    i.id = "constraints_0000"
    patch_image(monkeypatch, [i])
    cfg = {
        "run": "migration",
        "benchmark": "image",
        "thinking": "min",
        "prompt": "image",
        "models": [{"name": "Points", "provider": "mock", "model": "mock-points"}],
    }
    run(cfg, results_dir=tmp_path)
    path = tmp_path / "migration/mock-points/responses.jsonl"
    before = path.read_text()
    i.id = "image_0000"
    monkeypatch.setattr(runner, "get_client", lambda *args: pytest.fail("cached request called again"))
    rows = run(cfg, results_dir=tmp_path)
    assert rows[0]["n"] == 1
    assert path.read_text() == before


def test_account_quota_stops_new_calls(tmp_path, monkeypatch):
    from asimov import runner

    items = [item(tmp_path) for _ in range(4)]
    for n, i in enumerate(items):
        i.id = f"image_{n:04d}"
    patch_image(monkeypatch, items)
    calls = []

    class Client:
        def generate(self, *args, **kwargs):
            calls.append(1)
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(runner, "get_client", lambda *args: Client())
    cfg = {
        "run": "quota",
        "benchmark": "image",
        "thinking": "min",
        "prompt": "image",
        "models": [{"name": "Points", "provider": "mock", "model": "mock-points"}],
    }
    rows = run(cfg, workers=1, results_dir=tmp_path)
    assert len(calls) == 1
    assert rows[0]["n_missing"] == 4 and rows[0]["n"] == 0


def test_score_only_never_calls_provider(tmp_path, monkeypatch):
    from asimov import runner

    patch_image(monkeypatch, [item(tmp_path)])
    monkeypatch.setattr(runner, "get_client", lambda *args: pytest.fail("score-only made an API call"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = {
        "run": "offline",
        "benchmark": "image",
        "thinking": "min",
        "prompt": "image",
        "models": [{"name": "GPT-5", "provider": "openai", "model": "gpt-5"}],
    }
    rows = run(cfg, score_only=True, results_dir=tmp_path)
    assert rows[0]["n"] == 0 and rows[0]["n_missing"] == 1


def test_minute_limit_retry_hint_does_not_retry_daily_quota():
    from asimov.runner import minute_retry_delay

    class QuotaError(RuntimeError):
        details: dict

    error = QuotaError("quota")
    error.details = {
        "error": {
            "details": [
                {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"},
            ]
        }
    }
    assert minute_retry_delay(error) == 18
    error.details["error"]["details"][0]["violations"].append({"quotaId": "GenerateRequestsPerDay"})
    assert minute_retry_delay(error) is None


def test_image_integration_preserves_video_cache(tmp_path, monkeypatch):
    from asimov import runner
    from asimov.data.video import VideoAnswer, VideoItem

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    video = VideoItem("video_0000", (frame,), (0.0,))
    monkeypatch.setattr(
        runner,
        "get_benchmark",
        lambda name: replace(
            get_benchmark("video"),
            load_items=lambda: [video],
            load_answers=lambda: {video.id: VideoAnswer("No", -1, "A", "No Harm")},
        ),
    )
    cfg = {
        "run": "video",
        "benchmark": "video",
        "prompt": "video_v2",
        "thinking": "min",
        "models": [{"name": "No", "provider": "mock", "model": "mock-always-no"}],
    }
    rows = run(cfg, results_dir=tmp_path)
    assert rows[0]["risk_accuracy"] == 1
    path = tmp_path / "video/mock-always-no/responses.jsonl"
    before = path.read_text()
    assert json.loads(before)["key"] == runner.run_key("mock-always-no", get_benchmark("video").prompt_version("video_v2"), "min")
    monkeypatch.setattr(runner, "get_client", lambda *args: pytest.fail("video cache missed"))
    assert run(cfg, results_dir=tmp_path)[0]["n"] == 1
    assert path.read_text() == before


def test_system_split_preserves_exact_prompt_and_image(tmp_path):
    from asimov.bench.image import build_system_request
    from asimov.models.openai_client import to_input

    i = item(tmp_path)
    i.prompt = "You must obey <CONSTRAINTS START>Robot rules.<CONSTRAINTS END> \nPoint <1>. JSON please."
    system, parts = build_system_request(i)
    assert isinstance(parts[0], TextPart) and isinstance(parts[2], TextPart)
    assert isinstance(parts[1], ImagePart)
    assert system + parts[0].text + "<1>" + parts[2].text == i.prompt
    assert parts[1].data == i.image_path.read_bytes()
    messages = to_input(parts, system)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == system
    assert "Robot rules." not in messages[1]["content"][0]["text"]
    i.prompt = "No constraint block <1>"
    with pytest.raises(ValueError, match="constraint block"):
        build_system_request(i)


def test_message_roles_change_cache_key(tmp_path, monkeypatch):
    from asimov import runner
    from asimov.models.base import ModelResponse

    i = item(tmp_path)
    i.prompt = "Obey <CONSTRAINTS START>Rules.<CONSTRAINTS END> Point <1>."
    patch_image(monkeypatch, [i])
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    systems = []

    class Client:
        def generate(self, parts, thinking, system=None):
            systems.append(system)
            return ModelResponse("[]", "gpt-5-mini")

    monkeypatch.setattr(runner, "get_client", lambda *args: Client())
    cfg = {
        "run": "roles",
        "benchmark": "image",
        "thinking": "min",
        "prompt": "image",
        "models": [{"name": "Mini", "provider": "openai", "model": "gpt-5-mini"}],
    }
    run(cfg, results_dir=tmp_path)
    cfg["message_roles"] = "system_user"
    cfg["run"] = "roles_system"
    rows = run(cfg, results_dir=tmp_path)
    assert rows[0]["message_roles"] == "system_user"
    assert systems == [None, "Obey <CONSTRAINTS START>Rules.<CONSTRAINTS END>"]
    run(cfg, results_dir=tmp_path)
    assert len(systems) == 2
    from asimov.run_ids import run_dir

    old_path = run_dir(tmp_path, "roles", "image") / "gpt-5-mini/responses.jsonl"
    path = run_dir(tmp_path, "roles_system", "image") / "gpt-5-mini/responses.jsonl"
    saved = [json.loads(old_path.read_text()), json.loads(path.read_text())]
    assert saved[0]["key"] != saved[1]["key"]
    assert [r["message_roles"] for r in saved] == ["user", "system_user"]
