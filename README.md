# ASIMOV-2.0 reproduction and extensions

CS2881R AI safety mini-project. We reproduce the three result figures of Jindal et al.,
"Can AI Perceive Physical Danger and Intervene?" ([arXiv 2509.21651](https://arxiv.org/abs/2509.21651)),
then run four extension experiments. Due **2026-10-18**.

Read `docs/HANDOFF.md` for the full plan, and `WORKLOG.md` for every decision made so far.

## Quickstart

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env              # add your API keys; never commit .env
python scripts/download_data.py   # ~360 MB, idempotent
python -m pytest
python scripts/draw_boxes.py      # visual check of constraint boxes -> results/box_check/
```

## Build status

| Step | Status |
|---|---|
| 1. Download + convert (319 / 287 / 164 / 164) | done |
| 2. Constraints loader + tests + box overlay | done; **waiting on teammate visual check** |
| 3. Scorers | blocked on open decisions in WORKLOG (overlap rule, video metrics) |
| 4. Mock pipeline | not started |
| 5. Provider clients | not started |
| 6. Repro configs | not started |
| 7. Experiment READMEs | not started |

## Team

| | Week 1: reproduce (10/05 to 10/11) | Week 2: extend (10/12 to 10/16) |
|---|---|---|
| Person 1 (_name_) | Fig. 4, Injury | Exp 1: small-model prompting |
| Person 2 (_name_) | Fig. 5, Video | Exp 2: small-model training |
| Person 3 (_name_) | Fig. 6 (+ Fig. 7), Constraints | Exp 3: big models |
| Person 4 (_name_) | Starts one experiment early | Exp 4: classifiers |

10/17 to 10/18: combined figures and report.

## Layout

```
data/                  gitignored except data/README.md (schemas + real examples)
CLAUDE.md              rules and gotchas for coding agents
docs/HANDOFF.md        project brief
docs/presentation_outline.md  talk structure; owners fill in results
scripts/               download_data.py, draw_boxes.py (run.py, plot.py to come)
src/asimov/data/       tfrecord.py (raw reader), constraints.py (box parsing)
tests/
WORKLOG.md             dated decisions log
```
