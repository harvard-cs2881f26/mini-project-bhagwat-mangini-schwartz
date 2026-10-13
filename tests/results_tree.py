"""A small synthetic results/ tree for collect and plot tests: text + video runs and a mock run that is skipped."""

import json
from pathlib import Path

import yaml

ITEMS = ["t1", "t2", "t3", "t4"]
QUESTIONS = ("latent_risk", "latent_severity", "effect", "activated_risk")


def text_outcome(correct: list[bool], pred: str = "A") -> dict:
    out: dict = {q: {"pred": pred if ok else "B", "gold": pred, "correct": ok} for q, ok in zip(QUESTIONS, correct, strict=True)}
    out["parse"] = dict.fromkeys(QUESTIONS, "ok")
    out["parse_ok"] = True
    out["finish"] = "complete"
    return out


SCORED_AT = "2026-10-12T10:00:00-0400"


def write_model(
    run_dir: Path,
    name: str,
    model: str,
    provider: str,
    benchmark: str,
    per_item: dict,
    metrics: dict,
    scored_at: str = SCORED_AT,
) -> None:
    mdir = run_dir / model
    mdir.mkdir(parents=True)
    key = f"k-{model}"
    summary = {
        "name": name,
        "provider": provider,
        "model": model,
        "prompt": "p",
        "thinking": "min",
        "benchmark": benchmark,
        "run": run_dir.name,
        "key": key,
        "errors": 0,
        "input_tokens": 1000,
        "output_tokens": 100,
        "cached_input_tokens": 0,
        "truncation_rate": 0.0,
        "refusal_rate": 0.0,
        "n_with_reasoning": 0,
        "scored_at": scored_at,
        "git_commit": "abc1234def",
        "n": len(per_item),
        **metrics,
    }
    (mdir / "summary.json").write_text(json.dumps({**summary, "per_item": per_item}))
    rows = [
        {"id": i, "key": key, "served_model": f"{model}-2025", "latency_s": 1.0 + n, "text": f"reply {i}"}
        for n, i in enumerate(per_item)
    ]
    rows.append({"id": "t1", "key": "old-key", "served_model": "stale", "latency_s": 99.0, "text": "old"})
    (mdir / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def text_run(results: Path, run: str, large_correct: list[list[bool]], scored_at: str = SCORED_AT) -> None:
    run_dir = results / run
    models = [("GPT-5", "gpt-5", large_correct), ("GPT-5 Nano", "gpt-5-nano", [[False, True, False, True]] * 4)]
    cfg = {"run": run, "benchmark": "text", "models": [{"name": n, "provider": "openai", "model": m} for n, m, _ in models]}
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name, model, by_item in models:
        per_item = {i: text_outcome(c) for i, c in zip(ITEMS, by_item, strict=True)}
        metrics = {f"{q}_accuracy": sum(per_item[i][q]["correct"] for i in ITEMS) / 4 for q in QUESTIONS}
        metrics |= {"parse_failure_rate": 0.0, "latent_severity_n_severe": 2, "latent_severity_severe_recall": 0.5}
        write_model(run_dir, name, model, "openai", "text", per_item, metrics, scored_at)


def make_results(tmp_path: Path) -> Path:
    res = tmp_path / "results"
    text_run(res, "text_run", [[True, True, False, True]] * 4)
    vdir = res / "video_run"
    video_items = {
        f"v{i}": {
            "case": "error",
            "truth_s": 2.0,
            "pred_s": 2.0 + i,
            "error_s": float(i),
            "within_0_5": i == 0,
            "risk_ok": i != 2,
            "finish": "complete",
        }
        for i in range(3)
    }
    write_model(
        vdir,
        "GPT-5",
        "gpt-5",
        "openai",
        "video",
        video_items,
        {"risk_accuracy": 2 / 3, "within_0_5_rate": 1 / 3, "timing_mae_s": 1.0, "parse_failure_rate": 0.0},
    )
    write_model(res / "smoke", "Mock", "mock-1", "mock", "text", {"t1": text_outcome([True] * 4)}, {})
    (res / "comparisons.yaml").write_text("comparisons: []\n")
    return res
