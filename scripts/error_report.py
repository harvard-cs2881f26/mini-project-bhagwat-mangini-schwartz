#!/usr/bin/env python
"""Where models got the video benchmark wrong: error breakdowns plus example videos.

    python scripts/error_report.py configs/repro_fig5.yaml --models claude-sonnet-4-5-20250929 claude-opus-4-5-20251101

Reads saved replies only (no API calls). Writes results/error_review/<run>.html, which embeds
dataset frames, so it is gitignored. Numbers only: interpretation is left to the owner.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asimov.bench.video_score import parse_reply, snap_to_grid  # noqa: E402
from asimov.data.video import load_video_answers, load_video_items  # noqa: E402
from asimov.bench import get_benchmark  # noqa: E402
from asimov.run_ids import run_dir  # noqa: E402
from asimov.runner import load_saved, run_key  # noqa: E402

GROUPS = ["No, -1", "No, timed", "Yes"]
EXAMPLES = 6  # videos per gallery


def group(a) -> str:
    if a.risk == "Yes":
        return "Yes"
    return "No, -1" if a.last_intervention_s == -1 else "No, timed"


def truth_s(a) -> float:
    return -1 if a.last_intervention_s == -1 else snap_to_grid(a.last_intervention_s)


def thumb(path: Path, width: int = 200) -> str:
    im = Image.open(path)
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def table(head: list[str], rows: list[list]) -> str:
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    specs = [s for s in cfg["models"] if not args.models or s["model"] in args.models or s["name"] in args.models]
    A = load_video_answers()
    items = {i.id: i for i in load_video_items()}

    # replies[name][id] = (text, prediction or None)
    replies = {}
    for s in specs:
        saved = load_saved(run_dir(ROOT / "results", cfg["run"], cfg["benchmark"]) / s["model"] / "responses.jsonl",
                           run_key(s["model"], get_benchmark(cfg["benchmark"]).prompt_version(cfg["prompt"]), cfg["thinking"]))
        replies[s["name"]] = {i: (r["text"], parse_reply(r["text"])) for i, r in saved.items()}
    names = list(replies)
    ids = sorted(set.intersection(*(set(r) for r in replies.values())))

    def pred(n, i):
        return replies[n][i][1]

    def risk_ok(n, i):
        p = pred(n, i)
        return p is not None and p.risk == A[i].risk

    parts = [f"<h1>Error review: {html.escape(cfg['run'])} ({html.escape(cfg['prompt'])})</h1>",
             f"<p>{len(ids)} videos scored by every model below. Frames shown are every other frame (0.0, 1.0, … 7.0 s).</p>"]

    # 1. Risk accuracy by answer-key group
    rows = []
    for g in GROUPS:
        s = [i for i in ids if group(A[i]) == g]
        rows.append([g, len(s)] + [f"{sum(risk_ok(n, i) for i in s)} / {len(s)}" for n in names])
    parts += ["<h2>1. Risk answer correct, by answer-key group</h2>",
              "<p>“No, timed” = the key says no risk but still gives an intervention time.</p>",
              table(["Answer key", "Videos"] + names, rows)]

    # 2. Yes videos by severity and injury type
    rows = []
    for field in ("severity", "injury_type"):
        for v, c in sorted(Counter(getattr(A[i], field) for i in ids if A[i].risk == "Yes").items()):
            s = [i for i in ids if A[i].risk == "Yes" and getattr(A[i], field) == v]
            rows.append([field, v, c] + [f"{sum(risk_ok(n, i) for i in s)} / {c}" for n in names])
    parts += ["<h2>2. Risky videos caught, by severity and injury type</h2>",
              table(["Field", "Value", "Videos"] + names, rows)]

    # 3. Agreement
    if len(names) == 2:
        a, b = names
        c = Counter((risk_ok(a, i), risk_ok(b, i)) for i in ids)
        parts += ["<h2>3. Do the models fail on the same videos? (risk answer)</h2>",
                  table(["", f"{b} right", f"{b} wrong"],
                        [[f"{a} right", c[(True, True)], c[(True, False)]],
                         [f"{a} wrong", c[(False, True)], c[(False, False)]]])]

    # 4. Timing: signed error by when the true moment falls, and where predictions land
    rows = []
    for lo, hi, label in ((0, 2, "before 2 s"), (2, 5, "2–4.5 s"), (5, 9, "5 s or later")):
        r = [label]
        for n in names:
            e = [pred(n, i).time_s - truth_s(A[i]) for i in ids
                 if pred(n, i) and pred(n, i).time_s != -1 and lo <= truth_s(A[i]) < hi]
            r.append(f"n={len(e)}, mean {sum(e) / len(e):+.2f} s, |mean| {sum(map(abs, e)) / len(e):.2f} s" if e else "-")
        rows.append(r)
    parts += ["<h2>4. Timing error by when the true intervention moment is</h2>",
              "<p>Videos where both the key and the model give a time. Positive = model is late.</p>",
              table(["True time"] + names, rows)]
    grid = [-1.0] + [x / 2 for x in range(0, 17)]
    dist = [["answer key"] + [sum(truth_s(A[i]) == t for i in ids) for t in grid]]
    dist += [[n] + [sum(pred(n, i) is not None and pred(n, i).time_s == t for i in ids) for t in grid] for n in names]
    parts += ["<h3>Where the timestamps land (count of videos)</h3>", table(["", *[str(t) for t in grid]], dist)]

    # 5. Severity confusion on videos the key rates
    sev = ["No Harm", "Mild Harm", "Moderate Harm", "Severe Harm"]
    for n in names:
        rows = [[t] + [sum(1 for i in ids if A[i].risk == "Yes" and A[i].severity == t and pred(n, i)
                           and pred(n, i).severity == p) for p in sev] for t in sev[1:]]
        parts += [f"<h2>5. Severity: answer key (rows) vs {html.escape(n)} (columns)</h2>",
                  "<p>Risky videos with a rater consensus on severity.</p>", table(["Key \\ model", *sev], rows)]

    # 6. Galleries
    def biggest_timing(i):
        es = [abs(pred(n, i).time_s - truth_s(A[i])) for n in names
              if pred(n, i) and pred(n, i).time_s != -1 and truth_s(A[i]) != -1]
        return -min(es) if len(es) == len(names) else 0

    galleries = [
        ("Every model said Yes, key says No and -1",
         [i for i in ids if group(A[i]) == "No, -1" and all(not risk_ok(n, i) for n in names)]),
        ("Every model said Yes, key says No but gives a time",
         [i for i in ids if group(A[i]) == "No, timed" and all(not risk_ok(n, i) for n in names)]),
        ("Key says Yes, at least one model said No",
         [i for i in ids if A[i].risk == "Yes" and any(not risk_ok(n, i) for n in names)]),
        ("Largest timing error (every model gave a time)",
         sorted((i for i in ids if biggest_timing(i) < 0), key=biggest_timing)),
    ]
    for title, pool in galleries:
        parts.append(f"<h2>6. {html.escape(title)} ({len(pool)} videos; first {min(EXAMPLES, len(pool))} shown)</h2>")
        for i in pool[:EXAMPLES]:
            a = A[i]
            frames = "".join(f'<figure><img src="data:image/jpeg;base64,{thumb(p)}"><figcaption>{t:.1f} s</figcaption></figure>'
                             for p, t in list(zip(items[i].frame_paths, items[i].frame_seconds))[::2])
            key = f"risk {a.risk}, time {a.last_intervention_s:g} (scored as {truth_s(a):g}), severity {a.severity}, type {a.injury_type}"
            said = "".join(
                f"<details><summary><b>{html.escape(n)}:</b> "
                + (html.escape(f"{pred(n, i).risk}, {pred(n, i).time_s:g} s, {pred(n, i).severity}") if pred(n, i) else "unparseable")
                + f"</summary><pre>{html.escape(replies[n][i][0])}</pre></details>" for n in names)
            parts.append(f'<div class="video"><h3>{i}</h3><div class="frames">{frames}</div>'
                         f"<p><b>Answer key:</b> {html.escape(key)}</p>{said}</div>")

    css = """body{font:14px/1.45 system-ui,sans-serif;max-width:1300px;margin:24px auto;padding:0 16px;color:#111;background:#fff}
table{border-collapse:collapse;margin:8px 0 16px}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}th{background:#f3f3f3}
td:first-child,th:first-child{text-align:left}.frames{display:flex;gap:4px;overflow-x:auto}figure{margin:0}
figcaption{font-size:11px;color:#555;text-align:center}img{display:block;width:150px}.video{border-top:1px solid #ddd;padding:8px 0}
pre{white-space:pre-wrap;background:#f7f7f7;padding:8px;font-size:12px}summary{cursor:pointer}"""
    out = ROOT / "results" / "error_review" / f"{cfg['run']}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><meta charset=utf-8><title>Error review</title><style>{css}</style>" + "".join(parts))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
