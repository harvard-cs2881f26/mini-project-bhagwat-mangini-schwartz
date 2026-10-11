# CLAUDE.md

Guidance for Claude (and any coding agent) working in this repo.

## Project

CS2881R AI safety mini-project, four students, due **2026-10-18**. We reproduce Figs. 4, 5, 6
of Jindal et al., "Can AI Perceive Physical Danger and Intervene?" (ASIMOV-2.0, arXiv
2509.21651), then test two hypotheses:

- **H1:** small models can close the gap with large ones through prompting, training or other
  techniques. Judge them on accuracy *and* inference cost (latency, tokens, parameters, $).
- **H2:** newer LLM-based safety classifiers have a better accuracy/cost trade-off in these
  physical settings than general models do.

**Benchmark names:** we name the three benchmarks by input type, in prose and in code:
**text** (Fig. 4, paper: ASIMOV-2.0-Injury), **video** (Fig. 5), **image** (Fig. 6, paper:
ASIMOV-2.0-Constraints). Data folders, modules and configs use the same names. The paper's
dataset names appear only in `DATASETS` in `scripts/download_data.py`.

**Source of truth for tasks, owners and project docs:** the Google Doc "AI Safety - Mini Project"
(https://docs.google.com/document/d/1eh77clNa1WYOpOYLe7mxsv3JcFEXPAvDvj6xE79GYOo/edit, Drive file id
`1eh77clNa1WYOpOYLe7mxsv3JcFEXPAvDvj6xE79GYOo`). `docs/PROJECT.md` is its dated repo copy (team,
tasks, grading rubric, course guidance, presentation, open items). Before planning work or answering
"who owns X" / "what's left", re-read the Google Doc with the Drive connector. If it has changed,
update `docs/PROJECT.md` and its "last synced" date. If the doc and the repo disagree, the doc wins on
scope and owners. Point out the conflict instead of silently picking one.

`WORKLOG.md` is the technical decision log (methods). `docs/HANDOFF.md` is the original brief
(historical). `docs/presentation_outline.md` is the talk the results feed into.

## Hard rules

- **Do not write the experiments or the analysis.** The course penalizes AI-generated work
  the students can't explain. Build tools (loaders, scorers, clients, runners, plots). Leave
  hypotheses, methods, conclusions and slide text to the owners. When asked for analysis,
  give the numbers and point out what they show, but don't write the interpretation.
- **Ask before any run that costs money**, other than a `--limit N` dry run.
- **Cache every API response.** Reruns must be free. Never cache errors.
- **Log finished work in CONTRIBUTIONS.md** under the person it was done for, with an honest
  "AI assistance" entry saying what the AI did. Don't fill in someone's AI-assistance cell
  for work you didn't see; mark it for them to fill in.
- **Log every scoring decision in WORKLOG.md**, with the date, a status tag
  (DECIDED / PROPOSED / OPEN) and the alternative considered.
- **Never train or tune on eval items.** Training data for H1 must come from elsewhere.
- **Never commit keys or data.** `.env` and `data/*` (except `data/README.md`) are gitignored.
- **Keep the paper's prompts verbatim** for reproduction. Prompt variants go in `experiments/`.
- **Stop and ask** when the data contradicts the brief or a metric definition is ambiguous.
  Don't pick silently.

## Commands

```bash
source .venv/bin/activate              # Python 3.12, created with uv
python scripts/download_data.py        # GCS -> data/, idempotent, ~360 MB
python -m pytest                       # fast, no network, no API calls
python scripts/draw_boxes.py [ids...]  # overlay constraint boxes -> results/box_check/
```

## Layout

```
scripts/download_data.py      only code that reads TFRecords; writes data/<bench>/items.jsonl + media
scripts/draw_boxes.py         visual check of constraint boxes
scripts/run.py <config>       run any benchmark config; scripts/error_report.py is video-only
src/asimov/data/tfrecord.py   pure-Python TFRecord/tf.train.Example reader (no TensorFlow)
src/asimov/data/rows.py       shared: read data/<bench>/items.jsonl, parse answer-key JSON
src/asimov/data/<bench>.py    loaders: inputs and answers loaded by separate functions
src/asimov/data/image.py      also box parsing -> (ymin, xmin, ymax, xmax) on 0-1000
src/asimov/bench/__init__.py  Benchmark interface + registry (get_benchmark)
src/asimov/bench/parts.py     TextPart / ImagePart: provider-neutral request parts
src/asimov/bench/parse.py     shared: pull the JSON object out of a model reply
src/asimov/bench/<bench>.py   build_request + BENCHMARK; scorer in <bench>_score.py
src/asimov/models/            one client per provider + mock; get_client(provider, model)
src/asimov/runner.py          benchmark-agnostic: cache, calls, errors, scoring via the registry
configs/*.yaml                run, benchmark, prompt, thinking, models
tests/                        unit tests for our code (not model evals)
data/README.md                schemas with real examples, read before touching a loader
```

**Adding a benchmark** (text, image): write its loader, `build_request` and scorer, define
`BENCHMARK = Benchmark(...)` in `bench/<bench>.py`, add it to `MODULES` in `bench/__init__.py`,
and add a config. Don't edit the runner or the clients for one benchmark. Scorers stay separate:
they only share the return shape (`n`, metrics, `per_item`) and `bench/parse.py`.

Planned: `scripts/plot.py`, `experiments/*/README.md`.

## Data gotchas

- Image boxes are stored as **x, y on 0–1**. Models answer **[y, x] on 0–1000**. Only
  `to_yx1000` converts between them. Never convert anywhere else.
- Several raters boxed each object, so one object has about 5 boxes. Labels are free text
  and sometimes contradict the box's class. Score on geometry only.
- 43% of safe objects sit inside a violating box (e.g. sushi inside a violating bowl). The
  overlap rule is **OPEN** in WORKLOG. Don't hard-code one.
- **The video data has no eval prompt** (`prompt == "00:00.000"`), and the injury-type letters
  have no option text. Both must come from the paper appendix. Every video is 16 frames at 2 fps.
- Video `last_intervention_timestamp_seconds` is -1 for 98 items. 95 "No" items still have a
  timestamp. How the metrics treat these is **OPEN**.
- The two Image variants (`image`, `image_rationale`) are identical except for one sentence at
  the end of the prompt.

## Style

Small, readable Python with type hints and dataclasses. Match the existing modules. Keep
dependencies light (no TensorFlow). Tests check our code, not model quality.
