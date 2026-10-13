# Text extension notes (Part 2)

Working notes for Noa, kept for later. Not decided; nothing here is a hypothesis or a method yet.

## Which questions suit a decision model

Drafted by Claude on 2026-10-12. The "fit" column applies vendor guidance: decision models are
pitched for quick, one-step judgments between a few clearly described options, and their docs
list multi-step reasoning, "what happens after X" questions and ordered scales as weak spots.
The research behind it is in `reports/System One models accuracy cost.md`.
The accuracy column is the range over the 8 LLMs in `repro_fig4` (thinking
`min`), from each `results/repro_fig4/<model>/summary.json`.

| Question | What it asks | Fit for a decision model | Part 1 LLM accuracy (8 models) |
|---|---|---|---|
| Q1 injury type | pick 1 of 5 described injuries from the scenario | **best fit**: one step, distinct options | 0.86–0.93 (Nano 0.56) |
| Q2 latent severity | rate severity on a 4-level scale | medium: ordered scale, and depends on Q1's injury | 0.66–0.77 (Nano 0.53) |
| Q3 effect of instruction | imagine the robot acts; does risk go up or down? | weak: needs reasoning about what happens next | 0.56–0.69 (Nano 0.47) |
| Q4 activated severity | severity after the robot acts | weakest: needs reasoning about what happens next, and depends on Q1's injury | 0.60–0.67 (Nano 0.40) |

## Design options for using decision models (2026-10-13)

Options drafted by Claude, with a recommendation. **Not decided:** which ones Part 2 uses, and why,
go in `docs/HYPOTHESIS.md` in Noa's words. "Decision model" = a model that returns a probability
per option instead of writing text (Jev, luna Decisions, OpenJev, Laya, Kev, Qwen option logits,
NLI). The systems and their configs are in docs/PLAN.md (Session 3 Status row) and `configs/part2_*.yaml`.

| Option | What it is | Cost | Recommendation |
|---|---|---|---|
| **A. All questions, every model** | Every system answers all 4 questions on all 319 items; results per question | the runs themselves (~$0.55 hosted + local) | **Do.** Everything else is computed from it |
| **B. Split by question** | A decision model answers some questions (e.g. Q1-Q2), an LLM the rest (e.g. Q3-Q4); cost = the sum | $0, simulated from saved runs | **Do**, with the split fixed before looking at results, or every split reported as a table |
| C. Decision model's Q1 → LLM for Q2-Q4 | The decision model's injury pick goes into the LLM's prompt | new LLM prompt variant + paid LLM calls | Skip: most of it is covered by cumulative mode inside the decision models |
| **D. Route by confidence (cascade)** | The decision model answers; items where its top probability is below a threshold go to an LLM; accuracy vs cost as the threshold moves, plus a perfect-router upper bound | $0, simulated from saved runs | **Do**: the most direct accuracy-vs-cost answer |
| E. Recast Q3 as yes/no for safety classifiers | "Does the instruction increase injury risk?" for Llama Guard / ShieldGemma etc. | new models; a different task from the paper's | Skip unless the H2 owner wants it |

Related variants already built (one config line each): question modes `independent` vs
`cumulative` (each question sees the model's own earlier picks), reversed option order, OpenJev
think 0 vs 512, full vs short wording, letters vs option names (WORKLOG 2026-10-12/13).

**What B and D need**
- A price table (`configs/prices.yaml`, dated, with Gemini thinking tokens billed as output per
  WORKLOG "Cost method"), so each item's cost is in dollars. Local models: report latency and
  parameters instead of dollars.
- One analysis script reading only `results/` (e.g. `scripts/cascade.py`): D's curve and
  perfect-router bound, B's per-split table, accuracy with CIs from `src/asimov/stats.py`, cost
  per 1,000 items. Numbers and figures only.
- Deferral is per item (one LLM call answers all 4 questions). Confidence for an item = e.g. the
  lowest top-option probability across its 4 questions; the rule is a choice to log in WORKLOG.

**Choices for Noa before anyone looks at results** (log each in WORKLOG)
1. D: which LLM is the escalation target (e.g. GPT-5; or two, e.g. GPT-5 and GPT-5 Mini), and which
   decision model(s) go first.
2. D: the item-confidence rule, and whether a single threshold is reported (chosen on held-out
   folds) or only the whole curve.
3. B: the split to pre-register, or "report all splits".
4. The scoring question still OPEN: how Q2-Q4 are read when a model's Q1 injury differs from the key's.

**Guard rail:** choosing a split, threshold or model *because* it scored best on the 319 items is
tuning on the eval set (CLAUDE.md hard rule). Fix choices first, or report every option.
