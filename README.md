# ASIMOV-2.0 reproduction and extensions

CS2881R AI safety mini-project. We reproduce the three result figures of Jindal et al.,
"Can AI Perceive Physical Danger and Intervene?" ([arXiv 2509.21651](https://arxiv.org/abs/2509.21651)),
then run four extension experiments. Due **2026-10-18**.

We name the benchmarks by input type: **text** (Fig. 4, the paper's Injury), **video** (Fig. 5)
and **image** (Fig. 6, the paper's Constraints). Data folders, modules and configs use the same names.

## Source of truth

**[AI Safety - Mini Project (Google Doc)](https://docs.google.com/document/d/1eh77clNa1WYOpOYLe7mxsv3JcFEXPAvDvj6xE79GYOo/edit)**
is the single source of truth for tasks, owners and project documentation. Track and assign
work there.

**[`docs/PROJECT.md`](docs/PROJECT.md)** is the repo copy of it: team, tasks, grading rubric,
course guidance, presentation and open items, with the date it was last synced.

Other docs:
- `WORKLOG.md`: dated technical and scoring decisions (becomes the report's methods section)
- `data/README.md`: dataset schemas
- `docs/HANDOFF.md`: the original setup brief (historical)

## Quickstart

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env              # add your API keys; never commit .env
python scripts/download_data.py   # ~360 MB, idempotent
python -m pytest
python scripts/draw_boxes.py      # visual check of constraint boxes -> results/box_check/
```

## Running Fig. 5 (video)

```bash
python scripts/check_models.py                                        # free: which models each API serves
python scripts/run.py configs/smoke_mock.yaml                         # free: mock models, no keys
python scripts/run.py configs/repro_fig5.yaml --limit 1 --models gpt-5-nano   # 1 real call, reply printed
python scripts/run.py configs/repro_fig5.yaml --limit 5               # 5 videos per model with a key
python scripts/run.py configs/repro_fig5.yaml                         # all 287 (ask the team first: costs money)
```

Replies are saved to `results/repro_fig5/<model>/responses.jsonl` (committed, so the team shares
them) and reused on reruns. Metrics go to `summary.json` next to them. Growing `--limit` only calls
the new videos.

## Build status

| Step | Status |
|---|---|
| 1. Download + convert (319 / 287 / 164 / 164) | done |
| 2. Image loader + tests + box overlay | done; **waiting on teammate visual check** |
| 3. Scorers | video done; text and image not started (image blocked on overlap rule) |
| 4. Mock pipeline | done (`configs/smoke_mock.yaml`) |
| 5. Provider clients | done (Anthropic, OpenAI, Gemini, mock) |
| 6. Repro configs | `repro_fig5.yaml` done; fig4, fig6 not started |
| 7. Experiment READMEs | not started |

## Team and tasks

Tyler Mangini, Noa Schwartz, Riddhi Bhagwat. Owners and tasks are in the
[project doc](https://docs.google.com/document/d/1eh77clNa1WYOpOYLe7mxsv3JcFEXPAvDvj6xE79GYOo/edit),
mirrored in [`docs/PROJECT.md`](docs/PROJECT.md).
Timeline: reproduce 10/05 to 10/11, extend 10/12 to 10/16, combined figures and report 10/17 to 10/18.

## Layout

```
data/                  gitignored except data/README.md (schemas + real examples)
CLAUDE.md              rules and gotchas for coding agents
docs/PROJECT.md        repo copy of the project Google Doc (team, tasks, rubric, links)
docs/HANDOFF.md        project brief
docs/presentation_outline.md  talk structure; owners fill in results
scripts/               download_data.py, draw_boxes.py (run.py, plot.py to come)
src/asimov/data/       tfrecord.py (raw reader), image.py (box parsing)
tests/
WORKLOG.md             dated decisions log
CONTRIBUTIONS.md       who did what (graded); add a row when you finish something
```
