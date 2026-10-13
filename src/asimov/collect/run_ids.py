"""Run IDs for collect (by scoring order) and a generated one-sentence description of each run.

Pure: the I/O is in load.py and collect.py. The ID facts (prefixes, numbering, run_ids.yaml) are in asimov.run_ids.
results/run_ids.yaml maps run name -> id; IDs are never changed or reused.
Descriptions are never hand-written: `describe` builds one sentence from the run's own settings. It never refers
to other runs or to the run's position, so it stays true when runs are added or deleted. All wording is in the tables below.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from asimov.collect.load import RunData
from asimov.run_ids import next_number, prefix

NEVER_SCORED = datetime.max.replace(tzinfo=timezone.utc)  # a run without scored_at sorts last

# ---- wording: change it here -----------------------------------------------------------------
BENCHMARK_WORDS = {"text": "text", "video": "video", "image": "image", "image_rationale": "image"}
PROMPT_WORDS = {"text_verbatim": "the paper's prompt sent unchanged (text_verbatim)"}  # others: "the <p> prompt"
THINKING_WORDS = {
    "min": "min (the lowest reasoning each provider allows)",
    "default": "default (each provider decides how much to reason)",
}  # other values are just named
TEMPLATES = {
    "run": "{benchmark} benchmark, using {prompt}, with thinking set to {thinking}, on {models}.",
    "reasoning_logged": "Reasoning summaries are saved.",
}


@dataclass(frozen=True)
class RunLabel:
    """What the pages show for a run: its ID and generated description."""

    id: str
    description: str


@dataclass(frozen=True)
class Settings:
    prompt: str
    thinking: str
    log_reasoning: bool
    models: tuple[tuple[str, str], ...]  # (model id, config name), in the run's order


# ---- IDs -------------------------------------------------------------------------------------


def first_scored(run: RunData) -> datetime:
    """When the run's first model was scored (summary `scored_at`)."""
    stamps = [datetime.fromisoformat(s) for m in run.models if (s := m.summary.get("scored_at"))]
    return min(stamps, default=NEVER_SCORED)


def assign_ids(existing: dict[str, str], runs: Iterable[RunData]) -> dict[str, str]:
    """Every existing entry unchanged; each new run gets the next free number for its prefix, by first scored_at, then name."""
    ids = dict(existing)
    for run in sorted((r for r in runs if r.name not in existing), key=lambda r: (first_scored(r), r.name)):
        p = prefix(run.benchmark)
        ids[run.name] = f"{p}{next_number(ids, p)}"
    return ids


def missing_ids(existing: dict[str, str], runs: Iterable[RunData]) -> list[str]:
    return sorted(r.name for r in runs if r.name not in existing)


# ---- descriptions ----------------------------------------------------------------------------


def joined(values: Iterable[object]) -> str:
    return ", ".join(sorted({str(v) for v in values}))


def settings(run: RunData) -> Settings:
    """From the run's saved config.yaml when it has the setting, else from its model summaries."""
    config, summaries = run.config or {}, [m.summary for m in run.models]

    def pick(key: str, default: object = None) -> str:
        return str(config[key]) if key in config else joined(s.get(key, default) for s in summaries)

    if config.get("models"):
        models = tuple((m["model"], m.get("name") or m["model"]) for m in config["models"])
    else:
        models = tuple((s["model"], s.get("name") or s["model"]) for s in summaries)
    logged = config["log_reasoning"] if "log_reasoning" in config else any(s.get("log_reasoning") for s in summaries)
    return Settings(pick("prompt"), pick("thinking"), bool(logged), models)


def describe(run: RunData) -> str:
    """One plain sentence from the run's own settings, e.g. "Text benchmark, using the paper's prompt sent
    unchanged (text_verbatim), with thinking set to min (the lowest reasoning each provider allows), on 8 models."
    """
    s = settings(run)
    n = len(s.models)
    text = TEMPLATES["run"].format(
        benchmark=BENCHMARK_WORDS.get(run.benchmark, run.benchmark).capitalize(),
        prompt=PROMPT_WORDS.get(s.prompt, f"the {s.prompt} prompt"),
        thinking=THINKING_WORDS.get(s.thinking, s.thinking),
        models=f"{n} model{'' if n == 1 else 's'}",
    )
    return f"{text} {TEMPLATES['reasoning_logged']}" if s.log_reasoning else text


def run_labels(runs: list[RunData], ids: dict[str, str]) -> dict[str, RunLabel]:
    return {r.name: RunLabel(ids[r.name], describe(r)) for r in runs}
