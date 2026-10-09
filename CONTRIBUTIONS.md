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

## Summary by person

Current assignments, from the project doc. Replace them with a summary of finished work
before the writeup is submitted.

| Person | Assigned areas |
|---|---|
| Tyler Mangini | Project setup and repo, presentation template, Fig. 5 (video) |
| Noa Schwartz | Fig. 4 (text), Jev-style classifiers |
| Riddhi Bhagwat | Fig. 6 (constraints) |
