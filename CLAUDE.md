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

`docs/HANDOFF.md` is the original brief. `WORKLOG.md` is the decision log.
`docs/presentation_outline.md` is the talk the results feed into.

## Hard rules

- **Do not write the experiments or the analysis.** The course penalizes AI-generated work
  the students can't explain. Build tools (loaders, scorers, clients, runners, plots). Leave
  hypotheses, methods, conclusions and slide text to the owners. When asked for analysis,
  give the numbers and point out what they show, but don't write the interpretation.
- **Ask before any run that costs money**, other than a `--limit N` dry run.
- **Cache every API response.** Reruns must be free. Never cache errors.
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
src/asimov/data/tfrecord.py   pure-Python TFRecord/tf.train.Example reader (no TensorFlow)
src/asimov/data/constraints.py box parsing -> (ymin, xmin, ymax, xmax) on 0-1000
tests/                        unit tests for our code (not model evals)
data/README.md                schemas with real examples, read before touching a loader
```

Planned (see HANDOFF section 4): `src/asimov/{models,bench}/`, `runner.py`, `cache.py`,
`scripts/run.py`, `scripts/plot.py`, `configs/*.yaml`, `experiments/*/README.md`.

## Data gotchas

- Constraints boxes are stored as **x, y on 0–1**. Models answer **[y, x] on 0–1000**. Only
  `to_yx1000` converts between them. Never convert anywhere else.
- Several raters boxed each object, so one object has about 5 boxes. Labels are free text
  and sometimes contradict the box's class. Score on geometry only.
- 43% of safe objects sit inside a violating box (e.g. sushi inside a violating bowl). The
  overlap rule is **OPEN** in WORKLOG. Don't hard-code one.
- **The video data has no eval prompt** (`prompt == "00:00.000"`), and the injury-type letters
  have no option text. Both must come from the paper appendix. Every video is 16 frames at 2 fps.
- Video `last_intervention_timestamp_seconds` is -1 for 98 items. 95 "No" items still have a
  timestamp. How the metrics treat these is **OPEN**.
- The two constraints variants are identical except for one sentence at the end of the prompt.

## Style

Small, readable Python with type hints and dataclasses. Match the existing modules. Keep
dependencies light (no TensorFlow). Tests check our code, not model quality.
