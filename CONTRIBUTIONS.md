# Contributions

The rubric asks for "detailed group contributions in each writeup" and says "group members
should contribute equally" (see `docs/PROJECT.md`). This log is the record we'll write that
section from.

**How to use it**
- Add a row when you finish something. Keep rows short and concrete: what you made or
  decided, and where it lives (file, commit, slide, doc section).
- Fill in the **AI assistance** column honestly ("none", or what the AI did and what you
  did). The course penalizes AI slop and work students can't explain; a clear record of who
  reasoned about what protects everyone.
- Planned work and owners live in the [project doc](https://docs.google.com/document/d/1eh77clNa1WYOpOYLe7mxsv3JcFEXPAvDvj6xE79GYOo/edit).
  This file records work that's **done**.

## Log

| Date | Person | Contribution | Where | AI assistance |
|---|---|---|---|---|
| 2026-10-04 | Tyler | Provided the setup brief: project scope, team plan, build order and repo rules | `docs/HANDOFF.md` | _Tyler to fill in_ |
| 2026-10-04 | Tyler | Repo setup: data download and conversion for all four datasets, constraints box loader and tests, box-overlay script, data README | `scripts/`, `src/asimov/data/`, `tests/`, `data/README.md` | Code written by Claude (Claude Code) from Tyler's brief; Tyler reviewed |
| 2026-10-04 | Tyler | Found data issues that need team decisions: safe objects inside violating boxes (43%), video eval prompt missing from the dataset, unclear video timestamp ground truth | `WORKLOG.md` | Found by Claude during setup; flagged to Tyler for team decision |
| 2026-10-07 | Tyler | Defined the two hypotheses (H1 small models, H2 safety classifiers) and the presentation structure | `docs/presentation_outline.md` | Tyler set the structure and hypotheses; Claude drafted the outline |
| 2026-10-07 | Tyler | Presentation template: theme chosen from three options after two review rounds | [Deck](https://claude.ai/artifact/X4zHzrwjBHkvDCbv7uTRMj) | Claude designed the options and built the slides; Tyler chose the direction |
| 2026-10-08 | Tyler | Created the private GitHub repo and set up the project Google Doc as source of truth | GitHub, `docs/PROJECT.md` | Claude pushed the repo and wrote the repo copy |
| 2026-10-10 | Tyler | Fig. 5 reproduction design: read the paper's metric definitions; chose yes/no accuracy for 5a, timing penalties (record 3 metrics, report the paper's 2), answer-key rules (snap ties earlier, "No" videos with a timestamp keep it, 8.0 s kept) and prompt wording (no rationale, "potential" injury); emailed the first author | `WORKLOG.md`, `src/asimov/prompts/video_v1.txt` | Claude pulled the paper's figures and definitions, built a review page of answer-key oddities, and proposed options; Tyler made each decision |
| 2026-10-10 | Tyler | Video request builder and leakage test (B + H) | `src/asimov/data/video.py`, `src/asimov/bench/video.py`, `tests/test_video_request.py` | Code written by Claude; Tyler set the leakage requirement (review pending) |
| 2026-10-10 | Tyler | Model clients (C): Anthropic, OpenAI, Gemini and mock, plus a free availability check | `src/asimov/models/`, `scripts/check_models.py`, `tests/test_model_clients.py` | Code written by Claude; Tyler set the thinking and temperature policy (review pending) |
| 2026-10-10 | Tyler | Video scorer (E): answer-key snapping, the four timing cases, reported and analysis metrics, hand-worked tests; chose the Claude stand-ins | `src/asimov/bench/video_score.py`, `tests/test_video_score.py` | Code written by Claude from Tyler's scoring decisions (review pending) |
| 2026-10-10 | Tyler | Runner (F): config-driven runs with saved, reusable replies, staged --limit, and a free mock smoke config | `src/asimov/runner.py`, `scripts/run.py`, `configs/`, `tests/test_runner.py` | Code written by Claude to Tyler's spec (tight, runnable by Tyler; Claude ran mocks only) |
| 2026-10-11 | Noa | Benchmark naming by input type (text / video / image) across data folders, loaders, tests and docs; fixed the download script crashing with no arguments | `scripts/download_data.py`, `src/asimov/data/image.py`, `CLAUDE.md`, `data/README.md`, `WORKLOG.md` | Noa chose the naming rule (one axis, input type) and asked for it to match the code; Claude compared options, made the rename and checked tests and mock runs before/after (Noa reviewing) |
| 2026-10-11 | Noa | Shared benchmark interface so text, video and image run through one runner: registry, shared request parts, reply parser and row reader; video results verified unchanged | `src/asimov/bench/__init__.py`, `bench/parts.py`, `bench/parse.py`, `data/rows.py`, `src/asimov/runner.py`, `scripts/run.py` | Noa set the goal (reuse existing code, no custom text pipeline, DRY) and the review path (PR to Tyler); Claude designed the interface, wrote the code and ran the before/after checks (Noa reviewing) |
| 2026-10-12 | Noa | 5-item test sample per benchmark so end-to-end pipeline tests run without the full download; tests no longer skip silently; CLAUDE.md exception (max 5 samples, tests only) | `tests/fixtures/data/`, `scripts/make_fixtures.py`, `tests/conftest.py`, `tests/test_full_data.py`, `CLAUDE.md` | Noa chose committing a real sample (option A) and the max-5 rule; Claude laid out the options and license/size facts, wrote the script and test changes, and checked both modes (Noa reviewing) |

## Summary by person

Current assignments, from the project doc. Replace them with a summary of finished work
before the writeup is submitted.

| Person | Assigned areas |
|---|---|
| Tyler Mangini | Project setup and repo, presentation template, Fig. 5 (video) |
| Noa Schwartz | Fig. 4 (text), Jev-style classifiers |
| Riddhi Bhagwat | Fig. 6 (constraints) |
| 2026-10-10 | Tyler | Gemini stand-ins (2.5 retired for new keys) and Gemini 3 thinking levels; runner warns on unknown `--models` | `configs/repro_fig5.yaml`, `src/asimov/models/gemini_client.py`, `src/asimov/runner.py` | Code written by Claude from Tyler's stand-in decision (review pending) |
| 2026-10-10 | Tyler | Error review page for the video benchmark (breakdowns by answer-key group, timing by true time, severity confusion, example videos with frames and replies) | `scripts/error_report.py` | Script written by Claude at Tyler's request; numbers only, interpretation left to Tyler (review pending) |
