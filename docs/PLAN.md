# Fig 4 (text) reproduction + decision-model extension: plan

Owner: Noa Schwartz. Revised 2026-10-12 to fit the shared repo (`mini-project-asimov-v2`).
Suggested home: `docs/PLAN.md`. Every session reads `CLAUDE.md`, this file's **Status** table,
and the latest `WORKLOG.md` entries first.

Changes from the 2026-10-11 draft: this plan builds on the existing harness and doesn't add a
second one. It uses `WORKLOG.md` / `CONTRIBUTIONS.md` instead of a new PROGRESS.md, and keeps the
team CLAUDE.md.

**Four Claude sessions.** A new session starts only where Noa has to step in between (decisions
or review), or where one session would run out of context. Cost approvals happen *inside* a
session: Claude stops, shows the numbers, waits for a yes, and carries on.

---

## Status (each session updates its own row, nothing else)

| Step | Who | State | Evidence / link |
|---|---|---|---|
| Before 1: Part 1 decisions | Noa | done | thinking = `min`; all 8 `PAPER_MODELS`; severity axis = 4-way exact letter (decided early, in Session 1). WORKLOG 2026-10-12 |
| Session 1: build + run Fig 4 | Claude (Noa approves spend) | done, merged (PR #4) | [PR #4](https://github.com/gini007/mini-project-asimov-v2/pull/4); [PR #5](https://github.com/gini007/mini-project-asimov-v2/pull/5) merged; replies: 8 models × 319/319, 0 failed calls (`repro_fig4`); cost: $15.11 full run, about $15.36 for the session incl. 5-item runs (WORKLOG 2026-10-12); pytest: 168 passed on full data; 161 passed, 7 skipped on the 5-item sample; CI green |
| Session 2: stats + Fig 4 plot | Claude | done, merged (PR #8) | [PR #8](https://github.com/gini007/mini-project-asimov-v2/pull/8); pages: `results/README.md`, `results/txt0_repro_fig4/README.md`, `results/txt0_repro_fig4/paper_check.md`; figures: `figures/fig4a`, `4b`, `5a`, `5b` (paper + comparison); claims file reviewed and frozen (five matching reads, no published exact numbers); verifier: 495/496 match, 1 bootstrap bound within Monte Carlo noise (`results/verify/`); pytest 229 passed, 7 skipped; CI green incl. `collect.py --check`; stats methods DECIDED in WORKLOG |
| Optional: Weave export (Noa's runs) | Claude, after Noa checks dataset terms | todo | |
| Between 2 and 3: review, Part 2 scope, hypothesis | Noa (+ team) | Part 2 scope decided 2026-10-12; Part 1 review + hypothesis still to do | WORKLOG 2026-10-12 ("Part 2 approved") |
| Session 3: build decision-model systems | Claude (Noa approves smoke spend) | done, merged (PRs [#9](https://github.com/gini007/mini-project-asimov-v2/pull/9), [#17](https://github.com/gini007/mini-project-asimov-v2/pull/17), [#18](https://github.com/gini007/mini-project-asimov-v2/pull/18), [#20](https://github.com/gini007/mini-project-asimov-v2/pull/20), [#21](https://github.com/gini007/mini-project-asimov-v2/pull/21), [#22](https://github.com/gini007/mini-project-asimov-v2/pull/22)) | clients: `/v1/systemone` (Jev, OpenJev, Laya, Kev), OpenAI Decisions, local Qwen3.5-4B(-Base) logprob/generate and NLI on Modal; question modes independent vs cumulative; 32 Part 2 runs x 319/319 (txt2-txt33), every run checked for bugs (WORKLOG 2026-10-13/14); API $0.42 total (full runs $0.131 + variants $0.066 + grid $0.221), Modal GPU ~$10 (estimate from run times; exact charge on the Modal dashboard); bugs fixed: tie-breaking, probability-sum tolerance, bf16 logit ties (fp32 head), Modal queue latency (use `raw.remote.compute_s`); held for Noa: Qwen generate txt33 8.2% parse failures; pytest 566 passed (full data); CI green |
| Session 4: Part 2 runs + metrics | Claude (Noa approves spend) | analysis done; verifier matches (3,774/3,774); waiting for Noa to merge PRs #25, #26, #27 | **Confirmatory:** `results/part2_designs/` (rebuilt on main, unchanged), write-up `results/part2_designs/writeup.md`. **Exploratory:** router search (PR #26), prompt experiments $3.01 (PR #25). Verifier: `results/verify/part2/report.md`, 3,774 of 3,774 match (D/B tables 3,000; router-search single systems 730; 5 picks 44). Brier/ECE and `max_calls` DROPPED (§5). WORKLOG 2026-10-14 "Part 2 analysis, confirmatory vs exploratory" |
| After 4: write-up + slides | Noa | todo | |

---

## 1. Facts already established (don't re-derive)

- Data: `data/text/items.jsonl`, 319 rows, ids `text_0000…`, fields `id`, `prompt`, `answer_raw`.
  Format and label counts are in `data/README.md`.
- One prompt holds all 4 questions; options are lettered; answer is a fenced JSON of 4 letters:
  `latent_risk_response`, `latent_severity_response`, `effect_response`, `activated_risk_response`.
  **An LLM answers an item in 1 call**, same as a decision model.
- Q1 (risk type) has **5** options, different per item; E = "No realistic injury possible".
  Q3 order is A Eliminates, B Reduces, C Same (no impact), D Increases.
- Majority baselines (match paper App. D): severity 4-way 142/319 = 0.445; severe-vs-not
  180/319 = 0.564; effect 121/319 = 0.379; activated risk 108/319 = 0.339.
- The harness exists: `scripts/run.py <config>` → `asimov.runner` → `Benchmark` registry →
  `get_client`. Cache key = (item, model, prompt version, thinking), plus `log_reasoning` when it
  is on (`runner.run_key`). Errors never cached. Latency and tokens recorded per reply.
- Model stand-ins (Gemini 3.x for 2.5, Claude 4.5 for Opus 4.1 / Sonnet 4) are in
  `PAPER_MODELS`, logged in WORKLOG. Only the GPT-5 family is like-for-like.

## 2. Hard constraints (on top of CLAUDE.md)

- Paper prompt verbatim for Part 1. Variants live in `experiments/`.
- Unparseable reply = wrong; parse-failure rate always reported.
- **Ask Noa before any paid run** beyond `--limit N`. Show the projected cost from the limit run.
- Don't edit the runner or existing clients for one benchmark; if the runner must change
  (Session 3), it's a PR for Tyler.
- Claude produces code, tables, figures and run write-ups (what the results show, comparison
  with the paper, key takeaways).
- End of every session: update the Status row, add a `CONTRIBUTIONS.md` row (AI-assistance
  cell honest), log any scoring decision in `WORKLOG.md` (date, DECIDED/PROPOSED/OPEN,
  alternative considered).
- Follow the practices in §9 (tests first, one home per fact, clean code). `code-reviewer`
  checks diffs against them.

---

## 3. Part 1: reproduce Fig 4

### Before Session 1: Part 1 decisions (Noa, ~20 min)

Outcomes are in WORKLOG 2026-10-12 ("Fig. 4 (text) benchmark and run settings").
1. **Thinking:** DECIDED `min`, to match Fig 5. A `default` run can follow as its own config.
2. **Models:** DECIDED, all 8 from `PAPER_MODELS`. Gemini credits were topped up in Session 1.
3. **Severity metric:** DECIDED in Session 1 (planned for Session 2). The Fig 4 axis is 4-way
   exact-letter accuracy. Severe-vs-not accuracy and severe recall are kept as extras in every
   summary.
4. **Paper values:** not done yet. Re-reading Fig 4 off the paper, incl. GPT 5 Mini (0.72) below
   GPT 5 Nano (0.87) on risk type, is the Session 2 `paper-reader`'s job; Noa reviews.
5. **Where paper values live:** `configs/paper/claims.yaml` (§11), as planned. No outcome
   recorded yet; it's created in Session 2.

### Session 1: build and run Fig 4

**Step 0, tooling ($0). Done in Session 1.**
- `requirements-dev.txt` (ruff), `[tool.ruff]` and `[tool.pyright]` in `pyproject.toml`, and
  `pydantic` in `requirements.txt`.
- The hooks in `.claude/hooks/`, as described in §9.4. A live Write to `data/` was blocked, and the
  Stop hook ran.
- Still open, for the team to agree: the repo-wide `ruff format` (as a separate commit), and adding
  ruff/pyright to CI (`.github/workflows/`), which would change everyone's PRs.

**Step A, build ($0).** Following CLAUDE.md "Adding a benchmark":
- `src/asimov/data/text.py`: `load_text_items()` (id + prompt), `load_text_answers()`
  (4 letters per id, via `rows.read_answer_json`).
- `src/asimov/bench/text.py`: `build_request` = one `TextPart` with the verbatim prompt;
  `BENCHMARK`; register in `MODULES`. `prompt_version` follows the rule on
  `Benchmark.prompt_version` (it covers everything our code adds beyond the item): for the
  verbatim prompt that is nothing, so a named tag like `text_verbatim`. Any later suffix or
  wrapper must change it.
- `src/asimov/bench/text_score.py`: parse with `bench/parse.json_object`; per-question accuracy
  (4), severe-vs-not accuracy and severe recall for Q2 and Q4, parse-failure rate, `per_item`.
- `configs/repro_fig4.yaml` (models + thinking from the decisions), `configs/smoke_mock_text.yaml`.
- **One config per thinking setting.** `repro_fig4.yaml` is `thinking: min`, `run: repro_fig4`.
  A later `default` run gets its own `configs/repro_fig4_default.yaml` with
  `run: repro_fig4_default`, so the two summaries never overwrite each other (replies are already
  kept apart by the cache key). Don't flip `thinking:` inside an existing config.
- **Self-describing summaries (small runner change, PR for Tyler).** Each
  `results/<id>_<run>/<model>/summary.json` also records `benchmark`, `run`, the cache `key`, the
  config path and a hash of its contents, the git commit, a timestamp, and `n`. Rebuild the
  video summaries from saved replies (free) so old and new runs carry the same fields. Cost in $
  is added in Session 4 with the price table.
- **One run name = one config (same runner PR).** At the start of a run the runner writes
  `results/<id>_<run>/config.yaml`, an exact copy of the config used. If that file already exists with
  different contents, the runner stops with a message asking for a new `run:` name, so results
  from different settings can never mix under one name (§11).
- Tests: loader counts and keys, request = verbatim prompt, scorer on hand-made replies
  (correct, wrong, single-quoted JSON, missing key, garbage).
- Tests run on the committed 5-item sample (`tests/fixtures/data/text/`, used in CI via
  `ASIMOV_DATA_DIR`). Checks that need all 319 items (counts, majority baselines) go in
  `tests/test_full_data.py`, marked `full_data`.
- Check: `pytest` passes on the sample and on full data; `tests/test_leakage.py` passes for
  `text` (no answer or item ID in the request); the mock run writes 319 replies; a
  majority-answer check through the scorer reproduces the §1 baselines.

**Step B, limit run.** `python scripts/run.py configs/repro_fig4.yaml --limit 5`. Check parse
rate, read 2–3 replies, compute $/item from logged tokens × list price.
**Stop and show Noa:** model, parse rate, $/item, projected $ for 319. Wait for a yes.

**Step C, full run.** Rerun to retry errors (cached replies are free).

**In the background:** a `docs-reader` subagent per decision API writes `docs/apis/<name>.md`
(endpoint, auth, fields, limits, errors, example, UNVERIFIED marks) plus a ready/blocked table
of access and hardware needs. Reading docs is free and builds nothing, so it saves time in Session 3.

- **Subagents:** `run-monitor` (haiku) for the full run; `docs-reader` (background);
  `code-reviewer` on the diff before ending.
- **Done when:**
  - each model has 319 replies or is listed as blocked with the reason;
  - parse failure < 2% (otherwise stop and report, don't add a suffix to the paper prompt);
  - cost logged in WORKLOG;
  - API docs written. Done: `docs/apis/`.

**Built in Session 1 (reuse it, don't rebuild).** Details in WORKLOG 2026-10-12.
- `scripts/run.py <config> --no-calls`: rescores saved replies for free and keeps the last real
  run's error count.
- `results/<id>_<run>/<model>/errors.jsonl`: failed calls, never reused as replies (a rerun retries them).
- `src/asimov/models/reply_info.py`: `finish` per item (`complete` / `truncated` / `refused`);
  the runner adds `truncation_rate`, `refusal_rate` and `cached_input_tokens` per model.
- `log_reasoning` config flag: reasoning summaries go in a separate `reasoning` field, never
  scored; part of the cache key only when on (`runner.run_key`).
- `src/asimov/config.py`: pydantic configs (`RunConfig`, `ModelSpec`); extra keys fail at load.
- `results/<id>_<run>/config.yaml` per run: one run name = one config (`runner.ConfigChanged`).
- Self-describing `summary.json`: `benchmark`, `run`, `key`, `config_path`, `config_sha256`,
  `git_commit`, `scored_at`, `n`.
- Text benchmark: `data/text.py`, `bench/text.py`, `bench/text_score.py` (per item: pred, gold,
  correct, parse status incl. `near_key`, rationale present).
- `repro_fig5` summaries rescored with `--no-calls`; metrics identical to `main`.
- `docs/apis/`: decision-model API notes plus the ready/blocked table in
  [`docs/apis/README.md`](apis/README.md).
- Hooks in `.claude/hooks/` (Stop, PreToolUse, PostToolUse; wired in
  `.claude/settings.local.json`) and subagents in `.claude/agents/` (`code-reviewer`,
  `docs-reader`, `run-monitor`).

### Session 2: stats + Fig 4 plot ($0)

- `scripts/collect.py`: reads every `results/*/*/summary.json` and writes two tables,
  rebuilt from scratch each time (never hand-edited):
  - `results/index.csv`: one row per (run, model) with every field from the summary plus the
    metrics. This is how runs are found and compared, by column, not by folder name.
  - `results/per_item.csv`: run × model × item × question with `pred`, `gold`, `correct`,
    `parse_ok`. Every paired comparison (min vs default, small vs large, LLM vs decision model) is a
    join on item between two slices of it.
- `src/asimov/stats.py`: Wilson CI, paired bootstrap of accuracy differences, McNemar. Pure
  functions, tested on known values, fed from `per_item.csv`. (Session 4 adds Brier/ECE here.)
- `scripts/plot.py`: the paper's figures, rebuilt from `results/index.csv`, to the spec in §10.
  For each figure, two versions:
  - **paper-style**: same chart type, axes, axis titles, labels and legend as the paper, so ours
    and theirs can sit side by side on a slide;
  - **comparison**: ours vs the paper's values on the same axes (ours filled with 95% CI bars,
    paper hollow, majority baselines dashed where they apply).
  This session builds Fig 4a, 4b and 5a (5a needs the text run). Fig 5b uses Tyler's existing
  video results, so build it too, but tell Tyler since it's his figure. Fig 6 waits for the image
  scorer; leave a stub that says so. Output: `figures/<fig>_{paper,comparison}.png` + `.pdf`.
- `results/txt0_repro_fig4/table.csv`: ours vs paper with CIs; large-minus-small gaps with bootstrap
  CIs; 4a-vs-4b per model.
- **Results pages (§11).** `scripts/collect.py` also renders `results/README.md` (summary) and
  `results/<id>_<run>/README.md` (one page per run), and creates `results/headline.yaml` with the
  current runs filled in for owners to edit. Includes Tyler's video run. Hook it into the end of
  `scripts/run.py`, and add a `collect.py --check` step to the CI workflow (tell Tyler and Riddhi,
  since it affects their PRs).
- **Paper claims file (§11).** A `paper-reader` subagent re-reads the paper (arXiv HTML text and
  figures) and drafts `configs/paper/claims.yaml`: every number the paper reports for Figs 4-6
  (figure read-offs and numbers in the text), with figure/section and a short quote. Noa reviews
  it, then it's frozen (edit hook). `collect.py` renders the paper check from it after every run.
- Table and plot scripts select runs by fields in `index.csv` (e.g. `benchmark=text`,
  `thinking=default`), so a `repro_fig4_default` run, if one exists, can be compared per model on
  the same items (McNemar + paired bootstrap) with no new code.
- **Subagents:** `stats-verifier` recomputes every number from `responses.jsonl` with its own
  script (in `results/verify/`, not reading `stats.py`), reports mismatches > 0.001;
  `code-reviewer` on the diff.
- **Done when:** verifier says "all match"; figure and table exist. Claude lists the numbers
  for claims (a) close to paper, (b) small models lag, (c) actions harder, without
  interpreting them, and stops.

### Optional, end of Session 2: Weave export for Noa's results

A browsable view of Noa's replies (prompt, reply, parsed answer, correct or not, per model, side
by side) for error analysis. It's an extra view only: `results/` stays the source of truth.

- **Scope: Noa's runs only.** `benchmark=text` runs now (Fig 4, incl. `repro_fig4_default`) and
  the Part 2 runs later. Never video or image: those are Tyler's and Riddhi's, and frames are heavy.
- **Before the first upload, Noa checks** that the ASIMOV-2.0 dataset terms allow putting prompts
  on a third-party service, and logs the answer in WORKLOG. No upload until then.
- `scripts/export_weave.py`: reads `results/index.csv`, `per_item.csv` and each run's
  `responses.jsonl`, and logs each run × model as a Weave evaluation (one row per item: prompt,
  reply, parsed letters, gold, correct per question; plus the run's settings from its
  `config.yaml`). Use Weave's imperative evaluation-logging API; a `docs-reader` subagent writes
  `docs/apis/weave.md` from the current docs first.
- Re-exporting the same run is safe: rows are keyed by run, model and item, so nothing duplicates.
- Doesn't touch the runner, the clients, CI or anyone else's results. `weave` goes in
  `requirements-dev.txt`; `WANDB_API_KEY=` goes in `.env.example` (the key itself only in `.env`);
  the W&B project name lives in the script's config, e.g. `asimov-v2-noa`.
- Tests: the row-building part is a pure function, tested offline on the 5-item sample; the
  upload is not unit-tested.
- **Done when:** one Fig 4 run shows up in Weave with all 319 items per model, and the per-model
  accuracies in Weave match `index.csv` exactly. Link the Weave project from `results/README.md`.

---

## 4. Between Session 2 and Session 3 (Noa, with team for scope)

1. **Review Part 1.** Write your own notes (verdict on (a)–(c), open issues). If parsing is
   broken or numbers are off by > 10 points with no explanation, fix Part 1 first. Then log
   `PART 2 APPROVED: <date>` in WORKLOG.
2. **Pick conditions.** The rubric wants 1–2 extensions. Suggested core (the three contrasts):

   | id | system | contrast |
   |---|---|---|
   | `gpt5`, `gpt5nano` | Part 1 runs, reused | reference |
   | `qwen_gen` / `qwen_logprob` | Qwen3.5-4B-Base (Kev's base) option logits; the generate pairing is proposed in Session 3 | interface, small |
   | `luna_gen` / `luna_decide` | gpt-6-luna, Responses vs Decisions API | interface, frontier |
   | `kev4b` | Kev-4B on Modal vs `qwen_logprob` | trained-for-decisions |
   | `jev` | Jev via `/v1/systemone` | headline system |

   Optional, in cut order: `nli_zeroshot`, `semif`, `laya`, `openjev_think` (razorback16 on
   Codiv), `qwen_think`.
   Check overlap with the unowned "small model prompting / training" rows with the team.
3. **Wording rule:** decision models get (a) the paper's full rubric text per question, or
   (b) short instructions. (a) keeps inputs equal to the LLMs; (b) is what these APIs expect.
   Log the choice and the alternative.
4. **Hypothesis:** write `docs/HYPOTHESIS.md` (as H2 or a renamed H, to avoid clashing with
   CLAUDE.md's H1). Frozen before Session 4.
5. **Part 2 decisions** (DECIDED 2026-10-12, Noa; WORKLOG "Part 2 approved"):
   - `jev`: paid TypeSafe key (alternative: skip Jev). Jev 1.13 is documented as biased toward
     the first option, so option order may need permuting.
   - `kev4b`: Kev-4B on Modal (alternative: Kev-0.8B locally; Kev-4B doesn't fit this 16 GB Mac).
   - `openjev`: `razorback16/openjev`, hosted on Codiv, think 0 and 512 (alternatives:
     `Zefan-Cai/Open-Jev`; local, ruled out on the 16 GB Mac).
   - Logprob baseline: Qwen3.5-4B-Base, Kev's base (alternative: Qwen3-4B).
   - All optional systems kept, `semif` included.

   Still open:
   - `qwen_logprob` can't use the verbatim prompt (it asks for JSON first), so it needs a prompt
     variant in `experiments/`.
   - An explicit thinking level (`low|medium|high`, mapped per provider, e.g. Claude budget
     1024 / 4096) for Fig 7 / H1. Raise with Riddhi and Tyler.
6. **Experiment ideas: to-dos, not decided** (H1: can prompting close the small-model gap, and at
   what inference cost). Each is a variant under its own run name in `experiments/`. None of them
   changes the reproduction runs.
   - **Format-repair nudge** (`repro_fig4_repair`): when a reply isn't valid JSON, the runner sends
     one follow-up turn ("return the same answers as valid JSON only"). Measure:
     - how often it fixes the format;
     - whether answers change when nudged, not only the format;
     - the extra calls and cost.

     Needs multi-turn support in the clients (Tyler's code). Caveat: only models that slip get a
     second look.
   - **Structured output / JSON schema** (`repro_fig4_schema`): the API enforces the four answer
     keys and letters. This prevents invalid JSON and key slips entirely. Compare accuracy and
     cost against the verbatim run. It changes the request, so it's per-provider client work.
   - **Thinking `default` vs `min`** (`repro_fig4_default`): same items, paired comparison per model.

## 5. Part 2: decision models

### Session 3: build every decision-model system

Starts with: *check WORKLOG has a `PART 2 APPROVED:` line written by Noa. If not, stop.*

**Step A, decision interface (PR for Tyler).**
- Parse each item's verbatim prompt into scenario, instruction and the 4 questions with options
  (`data/text.py`). Test on all 319: every item parses, option counts are 5/4/4/4.
- Decision clients return the same reply shape LLMs do (JSON of 4 letters), so `text_score`
  is unchanged, plus `probs` per question saved in the response row.
- Wording for decision requests comes from one place (per the wording rule), never per client.
- Check: a mock decision client runs 319 items end to end; existing tests and the video run
  are unchanged.

**Step B, hosted clients.** `models/systemone.py`: one client for Jev / OpenJev (and Kev,
Laya if served), `base_url` + `model` from config. `models/openai_decisions.py` for Luna;
`luna_gen` uses the existing OpenAI client. Tests with mocked HTTP and fixtures from the API
docs: request snapshot; same wording across clients; probs sum to 1 and pred = argmax; option
order mapped by value; 429 vs 400 handling; refusal recorded as wrong and counted; a cache hit
makes no call.

**Step C, local models.** Heavy deps go in `requirements-local.txt`, never the main
requirements. `models/local_hf.py`: Qwen3.5-4B-Base (Kev's base; the generate pairing is proposed here) generate (thinking off) and logprob (softmax over
option letters; check each letter is one token). Kev wrapper. On 20 items, logprob argmax
should agree with thinking-off generation on most items.

**Step D, smoke.** `--limit 5` per system after asking Noa (cents). Log the served model
version. Profile local models: warm up, median of 3 passes on 50 items, record hardware →
`results/latency_local.csv`.

- **Subagents:** `code-reviewer` on the diff. If context runs low, finish the current step,
  update Status, and continue in a fresh session from the next step. Steps B and C touch
  separate files, so they can also go in two parallel worktrees if time is tight.
- **Done when:** all tests pass; each chosen system has smoked or is listed as blocked with
  the reason (no access, no GPU, broken install).

### Session 4: Part 2 runs + metrics

Rewritten 2026-10-14 to record what was actually done. The original plan (call cap, full runs,
Brier/ECE, reliability plots, contrast tables) is in git history.

**What was done**
- **Full runs:** folded into Session 3. Every Part 2 run (txt2-txt33) was run and merged there
  (PRs #17, #18, #20, #21, #22).
- **Price table:** `configs/prices.yaml`, dated, with Gemini thinking tokens billed as output
  (PR #19). The results pages show $ for text runs.
- **Confirmatory:** design D (route by confidence) and design B (split by question), pre-registered:
  - 4 decision models x GPT-5 / GPT-5 Mini;
  - `scripts/cascade.py`, `configs/analysis/part2_designs.yaml`, `results/part2_designs/` (PR #19);
  - choices DECIDED in WORKLOG 2026-10-13, "Part 2 design analysis, B and D";
  - write-up `results/part2_designs/writeup.md` (PR #27).
- **Exploratory:**
  - router search, cross-validated over items, all Part 2 runs (PR #26);
  - cheaper LLM targets (PR #19);
  - two paid prompt experiments feeding Jev's answers to GPT-5 / Mini, $3.01 (PR #25);
  - Qwen generate few-shot format run and a same-size comparison (in progress).
- **Cross-run write-up:** `docs/part2_writeup.md` (PR #24).
- **Verifier:** `stats-verifier` on the Part 2 tables: 3,774 of 3,774 match, 0 mismatches
  (`results/verify/part2/report.md`).

**Dropped** (Noa, 2026-10-14)
- **Brier/ECE tables and reliability plots:** the functions exist in `stats.py`, but no hypothesis or
  design needs them.
- **Per-run `max_calls` cap:** every paid run finished under projected cost with a person approving
  it, so it no longer protects anything.

- **Done when:** the verifier matches; PRs #25, #26 and #27 are merged by Noa.

### After Session 4: write-up + slides (Noa)

From `results/` and the figures. Fill the Fig 4 and H slides in the deck template.

---

## 6. Order and timeline (due 2026-10-18)

```text
[decisions] ─► Session 1 ─► Session 2 ─► [review + scope + hypothesis] ─► Session 3 ─► Session 4 ─► [write-up]
```

| Day | Work |
|---|---|
| Mon 10/12 | Decisions (incl. the severity axis, moved up from Session 2), Session 1 |
| Tue 10/13 | Session 2, review, scope, hypothesis |
| Wed 10/14 | Session 3 |
| Thu 10/15–Fri 10/16 | Session 4 |
| Sat 10/17–Sun 10/18 | Write-up and slides |

If Part 2 slips, present Part 1 with the extension as planned work. Cut order is in §4.

**Change (2026-10-12):** Part 2 was approved early, before Noa's Part 1 review, so Sessions 2
and 3 run in parallel. Session 4 is still gated on the Part 1 review and `docs/HYPOTHESIS.md`.

**Branches: one per session, off an up-to-date `main`, merged by PR before the next session.**
Each PR is one session's work, CI runs on it, and no branch stacks on an unmerged one.

| Session | Branch |
|---|---|
| 1 | `fig4-text-benchmark` |
| 2 | `fig4-stats-plot` |
| 3 | `decision-systems` (or `decision-hosted` + `decision-local` as parallel worktrees) |
| 4 | `part2-runs` |

**Open a draft PR right after the first commit** (`git push -u origin <branch>` then
`gh pr create --draft --base main`). CI (`.github/workflows/`) runs only on pull requests and
pushes to `main`, so a pushed branch without a PR gets no checks. With the draft PR open, every
push runs the tests. At the end of the session: CI green, then mark it ready
(`gh pr ready`) and request review.

Changes to Tyler's code (the runner) go in their own commits inside the session's PR so he
can review them separately. If he can't review in time, split them into a small PR of their own.

## 7. Opening prompt for each session

> Read `CLAUDE.md`, `docs/PLAN.md` (Status table and **Session <N>**) and the latest WORKLOG
> entries. We're doing **Session <N>**. Follow the practices in §9. Start in plan mode and propose how you'll meet its
> "Done when", which subagents you'll use and why, then wait for my approval. Before saying
> you're done, show the evidence, update the Status row, and add WORKLOG / CONTRIBUTIONS entries.

## 8. Subagents (`.claude/agents/*.md`, add only when first needed)

`code-reviewer` (read-only; reviews `git diff main...HEAD` against CLAUDE.md, §2 and §9:
data or answer keys edited, tests weakened, a fact from the §9.2 table defined twice, I/O in
pure functions, bare `except`, magic numbers, dead code, names that differ from §9.3; returns
findings by severity with file:line and a one-line fix), `run-monitor` (haiku; reads
`results/<id>_<run>/*/responses.jsonl`; returns rows done, parse %, errors by type, cost so far in
≤ 15 lines), `stats-verifier` (never reads `src/asimov/stats.py`), `docs-reader`, `skeptic`,
`paper-reader` (reads the paper only, writes a draft `configs/paper/claims.yaml`, marks every
value it can't read exactly as `approx: true`; never sees our results, so it can't be swayed by them).
High-severity reviewer findings are fixed before merging.

---

## 9. Practices

Carried over from the 2026-10-11 draft and adapted to this repo. The aim: adding a system or a
metric is one small change in one place, and a reviewer can trust every system was treated
the same.

### 9.1 Testing

- **Tests first.** In each build step, write the tests from that step's "Done when", run them,
  check they fail for the right reason, then write the code.
- **Never edit or delete a test to make it pass.** If a test looks wrong, stop and say why.
- **Unit tests stay offline and fast:** no network, no keys. HTTP is mocked with recorded
  responses in `tests/fixtures/`. Tests check our code, not model quality.
- **Evidence, not "should pass":** paste the pytest summary line, row counts and key numbers
  into the Status row before calling a step done.

### 9.2 One home for each fact

| Fact | Its only home | Everyone else |
|---|---|---|
| Paper prompt (Part 1) | `prompt` in `data/text/items.jsonl`, verbatim | `bench/text.py` sends it unchanged |
| Answer-key parsing | `data/rows.read_answer_json` | loaders call it |
| Reply parsing and scoring conventions | `bench/parse.json_object` + `bench/parse.locate_keys`; rules in CLAUDE.md "Scoring conventions (all benchmarks)" ([PR #5](https://github.com/gini007/mini-project-asimov-v2/pull/5)) | scorers call them |
| How a reply ended (`finish`), cached input tokens | `models/reply_info.py` | the runner records them; scorers don't re-read provider fields |
| Run config schema | `asimov/config.py` (`RunConfig`, `ModelSpec`) | the runner and scripts load configs through it |
| Question keys, letter→label maps, severe-vs-not collapse | constants at the top of `bench/text_score.py` | stats, plots and decision clients import them |
| Decision-model wording and options | one file, chosen in Session 3 (e.g. `configs/decision_questions.yaml`) | every decision client renders from it |
| Metric formulas (Wilson, bootstrap, McNemar, Holm, Brier, ECE) | `src/asimov/stats.py` | scorers and `plot.py` import; nothing re-derives them |
| Model ids, providers, thinking | `configs/*.yaml` | no literals in code |
| Prices | one table (e.g. `configs/prices.yaml`) | no literals in code |
| Paper's reported numbers (figures + text) | `configs/paper/claims.yaml` | plots, run pages and the paper check read it |
| Which runs exist and how each was made | `summary.json` per (run, model), gathered into `results/index.csv` by `scripts/collect.py` | comparisons select runs by its columns, never by parsing folder names |
| Cache, error handling, timing, saving replies | `asimov/runner.py` | clients only call the API and return `ModelResponse` |
| `/v1/systemone` protocol | `models/systemone.py` | Jev, OpenJev, Kev, Laya are config rows, not classes |
| Entry point | `scripts/run.py <config>` | no per-system scripts or notebooks |

- **Deliberate exception:** `stats-verifier` re-implements the metrics in `results/verify/`.
  Duplication is the point: code checking itself catches nothing. Say so in a comment there.
- **Rule of three:** no abstraction before the third concrete case. The systemone client
  qualifies; a generic "provider plugin framework" doesn't.

### 9.3 Clean code

- **Pure logic apart from I/O.** Parsing, scoring, label mapping and stats are pure functions
  (no files, network or clock), so tests run offline. I/O stays at the edges: loaders, the
  runner, clients, scripts.
- **Small units.** One responsibility per module; functions do one thing, roughly ≤ 40 lines.
  If a function needs comments to explain its sections, split it.
- **One set of names.** Question keys are the dataset's without `_response`: `latent_risk`,
  `latent_severity`, `effect`, `activated_risk`. The same strings appear in code, results
  columns and figures. (This replaces the draft's `risk_type` / `severity` / `action_effect`.)
  No abbreviations a reader has to decode.
- **Types.** Type hints everywhere; frozen dataclasses for items, answers and parsed configs.
  Configs are validated on load with pydantic models (`extra="forbid"`): an unknown key or
  missing field fails at startup, not mid-run. Done for run configs: `asimov/config.py`, with
  `pydantic` in `requirements.txt`.
- **Fail loudly and specifically.** No bare `except`; no swallowed errors. The runner's
  "print the error, don't save it, rerun to retry" stays, and every error is counted in the
  summary. Error messages say what to do next (e.g. "OpenJev rejected model 'jev-1.13.0';
  use 'jev-latest'"). Keep "no replies" distinct from "calls failed".
- **No magic values.** Thresholds (2% parse failure, 0.001 verifier tolerance, 90%
  logprob/SemIf agreement) are named constants or config entries.
- **No dead code.** No commented-out blocks, unused flags or `_old`/`_v2` files (prompt files
  like `video_v2.txt` are versions, not dead code). Exploration goes in a gitignored scratch
  folder, never in `src/`.
- **Deterministic.** Fixed seeds (item order, bootstrap), stable sorts, temperature left at
  provider default (WORKLOG). Every saved reply carries the served model id and the cache key.
- **Match the existing modules** (CLAUDE.md Style): small, readable, light dependencies.

### 9.4 Enforcement (hooks, not just prose)

All of this exists since Session 1 (scripts in `.claude/hooks/`).
- **Dev tools:** `requirements-dev.txt` with `ruff` (install:
  `pip install -r requirements-dev.txt`). The main `requirements.txt` stays runtime-only.
  pyright runs through `npx pyright` (Node is installed), so it needs no Python package.
  Settings are in `pyproject.toml` (`[tool.ruff]`, `[tool.pyright]`, basic mode).
- **PostToolUse on Edit|Write** (`track_edits.sh`): logs which files each session edits, so the
  Stop hook checks only that session's own work and never blocks another session in this folder.
  Edits made through Bash aren't seen, so code edits go through Edit/Write.
- **Stop hook** (`stop_check.sh`): runs only if the session edited `.py` files in this checkout.
  Then it runs `pytest -q` and `ruff check` on those files. `npx pyright --level error` and
  `ruff format --check` run only on the ones that are new on the branch. A session can't end with
  any of them failing.
- **pyright and format scope:** new files only. Existing modules (mostly Tyler's clients) don't pass
  yet; fixing them is a team decision, not a side effect of a session.
- **PreToolUse on Edit|Write** (`block_paths.sh`): blocks paths under `data/` (except
  `data/README.md`). Add `configs/paper/claims.yaml` once it's reviewed and `docs/HYPOTHESIS.md`
  once it's frozen.
- **First run of ruff format** will likely reformat existing files. Do it once, as its own
  commit agreed with the team, so it doesn't get mixed into a feature diff.
- **Where:** put these in `.claude/settings.local.json` (yours only). Moving them to
  `.claude/settings.json` affects Tyler and Riddhi, so ask them first. Check the syntax with
  `/hooks` in an interactive `claude` terminal.

### 9.5 Cost and runs

- `--limit 5` first, then a cost table, then wait for a yes (§2).
- Session 4 adds a per-run call cap (`max_calls`) before the paid runs; when it's hit the run
  stops and reports.
- Cost in $ comes from logged tokens × the price table (§9.2), never hand-typed.
- Batch API and prompt caching: skip unless the projected cost is high. The shared prefix of
  the text prompt is short, so caching saves little.
- If a step fails twice the same way, stop and report; don't loop.

### 9.6 Sessions and context

- Plan mode first; each session is a spec with an objective, inputs, out of scope, outputs,
  done-when, budget and subagents (§3–§5 follow it).
- Hand off through files (the Status table, WORKLOG), not chat.
- Delegate verbose, self-contained work (logs, doc reading, independent re-checks) to
  subagents; keep design in the main thread. Nesting ≤ 2 levels.
- Admit little into context: scripts print summaries; raw output stays on disk.
- Verifiers get the spec and the raw outputs, not the builder's reasoning.
- Report CIs, and cost and latency next to accuracy; a single run's score is noisy.

---

## 10. Paper figures to reproduce (checked against arXiv 2509.21651v2 on 2026-10-12)

Shared style for all of them: seaborn-like grey grid background, one fixed color per model
across every figure, a "Model" legend, and every point labeled with its model name. Stand-in
models are labeled with the paper name plus ours, e.g. "Claude Opus 4.1 (4.5)", from the
config's `name`. Accuracies are shown as 0–1 fractions (5b uses a 0–100 ratio, as in the paper).

| Fig | Type | x-axis | y-axis | Notes |
|---|---|---|---|---|
| 4a | scatter, 1 point per model | "Severity Accuracy (Higher is Better)": Q2 latent severity, 4-way | "Risk Recognition Accuracy (Higher is Better)": Q1 latent risk type | caption "Perceiving Latent Risks Types and Severity" |
| 4b | scatter | "Activated Risk Accuracy (Higher is Better)": Q4 | "Action Effect Accuracy (Higher is Better)": Q3 | caption "Understanding Action Effects" |
| 5a | grouped bars, text vs video per model, value printed on each bar | "Model" | "Risk Recognition Accuracy" | legend "Accuracy Type: Text / Video". The paper shows only Gemini 2.5 Pro, Claude Opus 4.1, GPT 5; plot those 3 for paper-style, all 8 for comparison. Text bar = Fig 4a y value. |
| 5b | scatter | "Last Time of Intervention Mean Absolute Error (Lower is Better, Inverted axis)": `timing_mae_s`, axis inverted | "Ratio of Accurate Interventions (Higher is Better)": `within_0_5_rate` × 100 | caption "Triggering Interventions" |
| 6a | horizontal bars, sorted, value printed at the bar end | "Constraint Violation Rate (%)", 0–80 | "Model" | caption "Adherence to Safety Constraints" |
| 6b, 6c | tables | Constraint Type | Violation Rate | top 5 constraint types for Opus 4.1 and for GPT 5 |
| 7 | small multiples, bars per thinking level | GPT: Minimal/Low/Medium/High; Claude: Low (1024)/High (4096) budget; Gemini: Off/On | "Violation Rate (%)" | Constraints only; Part 2 reference, not a reproduction target |

Paper values for the comparison versions come from `configs/paper/claims.yaml` (§11). The Fig 4 values in the original draft match the figure as checked on
2026-10-12, including GPT 5 Mini at 0.72 / 0.64.

---

## 11. Results pages and versioning

Goal: anyone on the team opens `results/README.md` on GitHub, sees the headline numbers, and can
click down to exactly how any number was produced.

```text
results/
  README.md                ← summary (generated): headline numbers per figure + list of all runs
  headline.yaml            ← hand-edited: which run is official per figure, plus notes
  comparisons.yaml         ← hand-edited: which runs to compare
  compare/<name>.md        ← comparison pages (generated)
  index.csv  per_item.csv  ← machine-readable tables
  run_ids.yaml             ← run name -> run ID (txt0, vid0, ...), added by the runner and collect.py
  <id>_<run>/              ← one folder per run, e.g. txt0_repro_fig4/ (asimov.run_ids.run_dir)
    README.md              ← run page (generated)
    config.yaml            ← exact config used, written by the runner at run start
    paper_check.md         ← ours vs every paper number this run can test (generated)
    <model>/responses.jsonl, summary.json
figures/
```

**Summary page (`results/README.md`).**
- One section per figure (4, 5, 6, then Part 2): the headline run's metrics per model with 95% CIs,
  the paper's value next to ours, the figure image, and a link to the run page.
- "All runs" table, newest first: run (link), benchmark, date, models, thinking, prompt version,
  git commit, cost, and the owner's note from `headline.yaml`.

**Run page (`results/<id>_<run>/README.md`).**
- Settings: benchmark, thinking, prompt version and a link to the prompt, git commit, date, link
  to `config.yaml`.
- Per model: requested vs served model id, stand-in note, replies / errors / parse failures,
  tokens, cost, latency p50/p95.
- Full metrics table with CIs, this run's figures, links to each `responses.jsonl` /
  `summary.json`, and links to the WORKLOG entries for the decisions it depends on.
- **Paper check** section (for reproduction runs): see below.

**Comparison pages (`results/compare/<name>.md`).** Listed by hand in `results/comparisons.yaml`
(e.g. `repro_fig4` vs `repro_fig4_default`); `collect.py` renders each one: per-question metrics
for both runs, the difference with a paired bootstrap CI and McNemar p-value, the overlaid figure,
and the items where they disagree (item id, each side's answer and a short reply excerpt inside
`<details>`). No prompt or scenario text in any page: that's dataset content we don't commit.

**Paper check (`results/<id>_<run>/paper_check.md`, linked from the run page and the summary).**
Rebuilt after every run from `configs/paper/claims.yaml`, which is written once by re-reading the
paper (Session 2) and re-read only if the paper changes (e.g. a new arXiv version).
- One row per paper claim that this run can test: claim id, figure/section, the paper's quote,
  paper value, our value with 95% CI, difference, and whether the paper value is inside our CI.
- Stand-in models are flagged on every row, since they aren't the paper's model.
- The paper's three headline claims (big models spot risks well, small models lag, actions are
  harder) appear as their computed checks (§3 claims a-c) with the numbers behind each.
- The mechanical checks, plus the run's write-up (`results/<id>_<run>/writeup.md`) on what they show.
- **Why not have an LLM re-read the paper on every run:** the paper doesn't change between runs,
  an LLM read costs money and can misread a number differently each time. Reading once into a
  reviewed file and comparing by code is free, repeatable and auditable.

**Rules.**
1. One run name = one config (the runner enforces it, Session 1). A change of settings means a
   new `run:` name, never an edit to an existing config.
2. Generated pages are never edited by hand, and nobody has to remember to rebuild them:
   - `scripts/run.py` calls `collect.py` at the end of every run, so pages update locally and get
     committed with the replies;
   - CI runs `python scripts/collect.py --check` on every PR: it rebuilds into a temp folder and
     fails if the committed pages differ ("results pages are stale: run scripts/collect.py").
   - `collect.py` reads only `results/`, never `data/` (scorers save `pred`, `gold`, `correct` per
     item in `summary.json`), so it works in CI, which has no dataset.
   - Runs themselves stay local: they need API keys and cost money, so they never run in CI.
3. Only `headline.yaml` and `comparisons.yaml` are hand-edited (plus `configs/paper/claims.yaml`
   once, at review): owners pick the official run per figure, choose comparisons, and write notes
   in their own words.
4. Each run page includes its write-up (`results/<id>_<run>/writeup.md`): what the results show, the comparison with the paper, and key takeaways.
