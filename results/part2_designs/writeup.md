## Part 2 designs D and B: pre-registered (confirmatory) result

Text benchmark, 319 items. Choices were fixed before analysis (WORKLOG 2026-10-13, "Part 2 design
analysis, B and D"). Everything is simulated from saved replies, at $0. Tables: `tables.md` in this folder;
figures: `figures/part2_designs/`. Exploratory work (other decision configurations, cheaper LLM targets,
the cross-validated router search, prompt experiments) is reported separately and is not part of this result.

### Summary
In the pre-registered analysis, no routing of the four decision models reached GPT-5's accuracy alone.
- **D (route by confidence):** the closest was Jev with its lowest-confidence 75% of items sent to GPT-5:
  mean-of-4 accuracy 0.723 [0.694, 0.752], vs 0.752 [0.723, 0.781] for GPT-5 alone. It cost $2.98 per
  1,000 items, vs $3.90.
- **D with GPT-5 Mini as the target:** Jev with 75% sent on gave 0.655 [0.625, 0.684] at $0.83, vs
  0.669 [0.639, 0.700] at $1.04 for Mini alone.
- **B, pre-registered split** (decision model answers Q1, the LLM Q2-Q4): below the LLM alone for every
  pairing, by 0.016 to 0.126 in mean-of-4 accuracy. Every paired-bootstrap CI excludes 0.

### Comparison with the LLMs alone
The paper has no decision-model analysis, so the comparison is with the LLMs alone, on the same 319 items.

**D, routed to GPT-5.** Mean-of-4 accuracy [95% CI], $ per 1,000 items:

| First model | Alone | 25% sent on | 50% sent on | 75% sent on | Perfect-router bound (items sent) |
|---|---|---|---|---|---|
| Jev | 0.581 [0.554, 0.607], $0.04 | 0.619, $1.04 | 0.695, $2.03 | 0.723 [0.694, 0.752], $2.98 | 0.792 (172) |
| luna Decisions | 0.515 [0.487, 0.542], $0.09 | 0.593, $1.07 | 0.663, $2.06 | 0.717 [0.687, 0.747], $3.02 | 0.787 (201) |
| OpenJev think 0 | 0.371 [0.344, 0.400], $0 (free tier) | 0.469, $0.99 | 0.571, $1.96 | 0.662 [0.629, 0.695], $2.93 | 0.776 (244) |
| Kev-4B | 0.457 [0.426, 0.488], GPU time | 0.545, $1.00 | 0.629, $1.99 | 0.697 [0.665, 0.731], $2.93 | 0.781 (223) |
| GPT-5 alone | 0.752 [0.723, 0.781], $3.90, p50 3.29 s | | | | |

**D, routed to GPT-5 Mini:** Mini alone scores 0.669 [0.639, 0.700] at $1.04. With 75% sent on, the four
first models give 0.655 (Jev), 0.655 (luna Decisions), 0.596 (OpenJev think 0) and 0.621 (Kev-4B), each
at $0.78-$0.87. The perfect-router bounds are 0.750, 0.722, 0.701 and 0.716.

**B, pre-registered split:** the split minus the LLM alone, mean of 4 [paired bootstrap CI]:

| First model | vs GPT-5 alone | vs GPT-5 Mini alone |
|---|---|---|
| Jev | -0.028 [-0.039, -0.018] | -0.016 [-0.027, -0.005] |
| luna Decisions | -0.028 [-0.039, -0.018] | -0.016 [-0.027, -0.006] |
| OpenJev think 0 | -0.107 [-0.121, -0.093] | -0.096 [-0.110, -0.082] |
| Kev-4B | -0.126 [-0.140, -0.111] | -0.114 [-0.129, -0.100] |

### Key takeaways
1. Alone, the four decision models score 0.371-0.581 in mean-of-4 accuracy, against GPT-5's 0.752. Jev is
   highest and costs $0.04 per 1,000 items, about 1% of GPT-5's $3.90.
2. In D, accuracy rises with the share of items sent to GPT-5 for every first model. At 75% sent on, Jev
   reaches 0.723 at $2.98: 0.029 below GPT-5 alone, at 76% of its cost. The two CIs overlap, but no paired
   test of that gap was pre-registered.
3. Latency does not fall. With 50% or more sent on, p50 latency is 2.2-3.5 s (calls in sequence), close to
   GPT-5's 3.29 s.
4. The perfect-router bounds (0.776-0.792 with GPT-5) sit above GPT-5 alone. Confidence routing at 75%
   reaches 0.662-0.723.
5. B cannot save money in this simulation, because one LLM call still answers all 4 questions. Its
   pre-registered split loses accuracy in every pairing: Jev and luna answer Q1 correctly on 0.821 of items,
   GPT-5 on 0.934.

### Caveats
- One benchmark (319 items), one sample of replies per system, thinking `min` for the LLMs.
- Kev-4B's $ leaves out its Modal GPU time. OpenJev's $0 reflects Codiv's free tier, not a market price.
- Latency assumes the two calls run in sequence; running them in parallel is in the exploratory analysis.
- The perfect router is a bound, not a system: it needs the answer key to choose.
- The simulation rules were confirmed by Noa (DECIDED 2026-10-14; WORKLOG "Rules for the simulation"):
  - a missing probability gives confidence 0;
  - ties at a fixed share are broken by item id;
  - the bound is computed at equal items sent;
  - a routed item pays for both calls, and its latency is their sum.
- Strict scoring (against the answer key). Q2-Q4 accuracy on items with Q1 matched is in each run's
  `summary.json`, not in these tables.

## EXPLORATORY: not part of the pre-registered result

Chosen after the Part 2 results were seen. Read these as leads, not evidence for H1/H2.

### Router search (PR #26, `results/router_search/`)
- **What it searched:** 30 decision-model configurations (system x wording x mode, plus the local models)
  and 18 LLMs, combined 8 ways, 60,078 candidates in all. Each pick is chosen on 4/5 of the items and scored
  on the remaining 1/5 (5 folds x 5 seeds). Numbers below are held out; in-sample picks were 0.003-0.030
  higher.
- **Held-out picks.** Mean of 4 [CI], $ per 1,000 items, p50 latency, most-picked candidate:

| Cap | Mean of 4 [CI] | $ per 1,000 | p50 | Most picked (share of folds) |
|---|---|---|---|---|
| $0.10 | 0.728 [0.698, 0.758] | $0.07 | 4.94 s | OpenJev think 512 full, lowest 10% to Gemini Flash-Lite [default] (40%) |
| $1 | 0.733 [0.706, 0.759] | $0.82 | 1.58 s | luna Decisions letters + Jev full cumulative, GPT-5 [min] when they disagree on Q1 (32%) |
| $2 | 0.759 [0.732, 0.786] | $1.69 | 1.95 s | Jev full cumulative, lowest 50% to GPT-5 [min], per-question mix (36%) |
| $4 and up | 0.780 [0.752, 0.808] | $3.10 | 3.88 s | the same at 80% (80%) |
| p50 ≤ 1 s | 0.774 [0.746, 0.802] | $2.09 | 0.82 s | Jev full cumulative, lowest 50% to GPT-5 [min], per-question mix (80%) |
| GPT-5 [min] alone | 0.752 [0.723, 0.781] | $3.90 | 3.29 s | |
| GPT-5 Mini [min] alone | 0.669 [0.639, 0.700] | $1.04 | 2.89 s | |

- **Self-hosted models** (Kev, Qwen, NLI) were never picked among all candidates at any cap.

### Prompt experiments (PR #25, `results/prompt_experiments/`; paid, $3.01)
Jev's saved answers were fed to the LLM. Mean of 4 minus the LLM alone [paired bootstrap CI]:

| System | Mean of 4 | vs LLM alone | $ per 1,000 |
|---|---|---|---|
| GPT-5 shown Jev's 4 answers | 0.729 | -0.023 [-0.043, -0.003] | $4.21 |
| Jev's Q1 + GPT-5 answers Q2-Q4 | 0.680 | -0.071 [-0.101, -0.043] | $3.39 |
| GPT-5 Mini shown Jev's 4 answers | 0.623 | -0.046 [-0.072, -0.020] | $1.14 |
| Jev's Q1 + GPT-5 Mini answers Q2-Q4 | 0.637 | -0.032 [-0.060, -0.004] | $0.87 |

**Caveats:**
- The prompt wording was confirmed by Noa (DECIDED 2026-10-14).
- The variant runs ran in parallel, so their latency can't be compared with `repro_fig4`'s.
