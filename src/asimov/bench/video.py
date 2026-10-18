"""Video benchmark: build the model request for one video.

A request is a provider-neutral list of parts, in order:
    text   prompt up to {FRAMES}
    16 x  (text "t = 1.5 s", image JPEG bytes)
    text   prompt after {FRAMES}
Each provider client turns these parts into its own message format. Images are sent unchanged
(full resolution, original JPEG bytes) and unnamed: no video ID or file name is ever sent.

Decision prompts (`decision_<llm prompt>`, e.g. decision_video_v2) send one DecisionPart instead,
for the decision classifiers. Same words as the LLM prompt, read from its file: the text before
{FRAMES} is the state, the 16 labelled frames follow it, then "Answer three questions...". Each
"Qn." paragraph, verbatim, is one choice question's instructions, with fixed options in the prompt's
own words (Yes/No; No/Mild/Moderate/Severe Harm; -1 or a frame timestamp). Only the closing "Respond
with only this JSON object" block is dropped: the classifiers return typed answers instead. Their
answers come back under the LLM reply's keys, so `video_score` scores both the same way.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from asimov.bench import Benchmark
from asimov.bench.parts import DecisionPart, DecisionQuestion, Frame, ImagePart, Part, TextPart
from asimov.bench.video_score import score
from asimov.data.video import VideoItem, load_video_answers, load_video_items

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_PROMPT = "video_v2"


def load_prompt(name: str = DEFAULT_PROMPT) -> str:
    template = (PROMPTS_DIR / f"{name}.txt").read_text()
    if template.count("{FRAMES}") != 1:
        raise ValueError(f"prompt {name} must contain {{FRAMES}} exactly once")
    return template


def frame_label(seconds: float) -> str:
    return f"t = {seconds:.1f} s"


DECISION_PREFIX = "decision_"
# (reply key, option values); each option's description is its own value. The 17 timing options are
# -1 and the frame grid, which is also the answer key's grid (video_score snaps key times to it).
SEVERITY_OPTIONS = ("No Harm", "Mild Harm", "Moderate Harm", "Severe Harm")
TIME_OPTIONS = ("-1", *(f"{i / 2:.1f}" for i in range(16)))
DECISION_QUESTIONS = (
    ("risk_of_physical_injury", ("Yes", "No")),
    ("injury_severity", SEVERITY_OPTIONS),
    ("last_intervention_timestamp_seconds", TIME_OPTIONS),
)


VARIANTS_PATH = Path(__file__).resolve().parents[3] / "experiments" / "video_decision_variants" / "variants.yaml"


def split_decision_name(name: str) -> tuple[str, str | None]:
    """'decision_video_v2_defs' -> ('video_v2', 'defs'); 'decision_video_v2' -> ('video_v2', None)."""
    rest = name.removeprefix(DECISION_PREFIX)
    if (PROMPTS_DIR / f"{rest}.txt").exists():
        return rest, None
    base, _, variant = rest.rpartition("_")
    if variant not in load_variants():
        raise ValueError(f"unknown video decision prompt {name}; variants: {sorted(load_variants())}")
    return base, variant


def load_variants() -> dict:
    return yaml.safe_load(VARIANTS_PATH.read_text())


def prompt_version(name: str) -> str:
    """Saved-reply key text: the prompt file's text, marked for decision prompts so their replies never
    mix with the LLM prompt's; a variant adds its own spec."""
    if name.startswith(DECISION_PREFIX):
        base, variant = split_decision_name(name)
        spec = "" if variant is None else json.dumps(load_variants()[variant], sort_keys=True) + "|"
        return "decision|" + spec + load_prompt(base)
    return load_prompt(name)


def decision_wording(prompt_name: str) -> tuple[str, str, list[str]]:
    """(state, lead, [Q1, Q2, Q3] instructions) from the LLM prompt file, verbatim."""
    before, after = load_prompt(prompt_name).split("{FRAMES}")
    paras = [p.strip() for p in after.split("\n\n") if p.strip()]
    questions = [p.split(". ", 1)[1] for p in paras if p[:3] in ("Q1.", "Q2.", "Q3.")]
    leads = [p for p in paras if not p.startswith(("Q1.", "Q2.", "Q3.", "Respond", "{"))]  # the JSON block is dropped
    if len(questions) != 3 or len(leads) != 1:
        raise ValueError(f"prompt {prompt_name}: expected one lead paragraph and Q1-Q3, got {len(leads)} and {len(questions)}")
    return before.strip(), leads[0], questions


def build_decision_request(item: VideoItem, prompt_name: str, variant: str | None = None) -> list[Part]:
    """The decision request; `variant` (experiments/video_decision_variants/variants.yaml) changes only
    what that variant names: option descriptions (defs), a first injury question (injury), frames (1fps)."""
    state, lead, instructions = decision_wording(prompt_name)
    spec = load_variants()[variant] if variant else {}
    descriptions = spec.get("descriptions", {})
    seconds, paths = list(item.frame_seconds), list(item.frame_paths)
    values_by_key = dict(DECISION_QUESTIONS)
    if "frame_step" in spec:
        seconds, paths = seconds[:: spec["frame_step"]], paths[:: spec["frame_step"]]
        state = spec["intro"]
        values_by_key["last_intervention_timestamp_seconds"] = ("-1", *(f"{s:.1f}" for s in seconds))
    questions = [
        DecisionQuestion(
            id=key, instructions=text, letters=values_by_key[key], reply_key=key,
            options=tuple((v, descriptions.get(key, {}).get(v, v)) for v in values_by_key[key]),
        )
        for (key, _), text in zip(DECISION_QUESTIONS, instructions, strict=True)
    ]
    mode = "independent"
    if "question" in spec:
        q = spec["question"]
        none_value, none_text = q["none_option"]
        options = ((none_value, none_text), *((o, o) for o in q["options"]))
        first = DecisionQuestion(id=q["id"], instructions=q["instructions"], options=options,
                                 letters=tuple(v for v, _ in options), reply_key=q["id"], context=q["context"])
        questions, mode = [first, *questions], "staged"
    frames = tuple(Frame(frame_label(s), p.read_bytes()) for s, p in zip(seconds, paths))
    return [DecisionPart(state=state, questions=tuple(questions), frames=frames, lead=lead, mode=mode)]


def build_request(item: VideoItem, prompt_name: str = DEFAULT_PROMPT) -> list[Part]:
    if prompt_name.startswith(DECISION_PREFIX):
        return build_decision_request(item, *split_decision_name(prompt_name))
    before, after = load_prompt(prompt_name).split("{FRAMES}")
    parts: list[Part] = [TextPart(before.rstrip("\n"))]
    for seconds, path in zip(item.frame_seconds, item.frame_paths):
        parts.append(TextPart(frame_label(seconds)))
        parts.append(ImagePart(path.read_bytes()))
    parts.append(TextPart(after.lstrip("\n")))
    return parts


BENCHMARK = Benchmark(
    name="video",
    load_items=load_video_items,
    load_answers=load_video_answers,
    build_request=build_request,
    score=score,
    prompt_version=prompt_version,  # the prompt file's text, so editing a prompt file invalidates saved replies
    columns=(
        ("risk acc", "risk_accuracy", True),
        ("avg err s", "timing_mae_s", False),
        ("≤0.5s", "within_0_5_rate", True),
        ("miss", "miss_rate", True),
        ("false al", "false_alarm_rate", True),
        ("parse fail", "parse_failure_rate", True),
    ),
)
