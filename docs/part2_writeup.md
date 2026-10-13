# Part 2: decision models on the text benchmark

Cross-run write-up of the Part 2 systems (runs txt2-txt33). The design analysis (cascade D, split B)
has its own write-up in `results/part2_designs/`. Numbers come from `results/index.csv` and the run
pages; accuracy is exact-letter accuracy on 319 items with 95% Wilson intervals in brackets; "mean" is
the plain mean of the four questions. Latency is p50 per item (Modal rows: `raw.remote.compute_s`;
cumulative rows: the sum of 4 calls). Cost is API $ for 319 items; Codiv rows are free tier, Kev and
the local rows are self-hosted (GPU time, no API price).

### Summary

No decision model, in any setting, matched GPT-5 on its own: the best ones came within a few points on
the mean but stayed below it on Q1. At its best setting, Jev scored a mean of 73.2% against GPT-5's
75.2% for $0.038 instead of $1.24 per 319 items, and in 0.70 s instead of 3.29 s per item. The same
setting choices moved results far more than the choice of system: wording and question mode alone took
Jev from 58.1% to 73.2% and OpenJev from 37.1% to 64.7%. The best cells were picked after running the
whole grid, so they are exploratory until the cross-validated analysis confirms them.

| System (setting) | Q1 risk type | Q2 severity | Q3 effect | Q4 activated | Mean | Parse fail | p50 latency | Cost |
|---|---|---|---|---|---|---|---|---|
| GPT-5, thinking min (txt0) | 93.4 [90-96] | 77.4 [73-82] | 64.3 [59-69] | 65.5 [60-71] | 75.2 | 0.0% | 3.29 s | $1.24 |
| GPT-5 Mini, thinking min (txt0) | 88.7 [85-92] | 70.2 [65-75] | 56.1 [51-61] | 52.7 [47-58] | 66.9 | 0.0% | 2.89 s | $0.33 |
| GPT-5 Nano, thinking min (txt0) | 55.5 [50-61] | 53.3 [48-59] | 46.7 [41-52] | 40.4 [35-46] | 49.0 | 0.3% | 2.69 s | $0.068 |
| Jev, full wording, cumulative (txt15) | 86.5 [82-90] | 68.0 [63-73] | 74.9 [70-79] | 63.3 [58-68] | 73.2 | 0.0% | 0.70 s | $0.038 |
| OpenJev think 512, full wording (txt23) | 86.2 [82-90] | 66.5 [61-71] | 75.2 [70-80] | 63.0 [58-68] | 72.7 | 0.0% | 4.92 s | free tier |
| gpt-6-luna generate (txt7) | 83.4 [79-87] | 69.0 [64-74] | 64.3 [59-69] | 57.7 [52-63] | 68.6 | 0.0% | 2.66 s | $0.073 |
| luna Decisions, full wording, cumulative (txt18) | 82.4 [78-86] | 65.2 [60-70] | 62.7 [57-68] | 60.8 [55-66] | 67.8 | 0.3% | 0.52 s | $0.062 |
| OpenJev think 0, full wording (txt14) | 78.4 [74-83] | 67.1 [62-72] | 68.3 [63-73] | 45.1 [40-51] | 64.7 | 0.0% | 0.45 s | free tier |
| Qwen3.5-4B generate, 2048 cap (txt33) | 77.1 [72-81] | 61.8 [56-67] | 59.6 [54-65] | 58.0 [53-63] | 64.1 | 8.2% | 16.77 s | self-hosted |
| Kev-4B, full wording (txt30) | 58.0 [53-63] | 59.6 [54-65] | 64.6 [59-70] | 40.1 [35-46] | 55.6 | 0.0% | 0.46 s | self-hosted |
| Qwen3.5-4B-Base, one-shot (txt29) | 80.9 [76-85] | 53.9 [48-59] | 51.4 [46-57] | 35.7 [31-41] | 55.5 | 0.0% | 0.23 s | self-hosted |
| NLI DeBERTa-v3-large (txt25) | 60.8 [55-66] | 36.4 [31-42] | 32.3 [27-38] | 24.8 [20-30] | 38.6 | 0.0% | 0.15 s | self-hosted |
| Laya, short wording (txt2) | 23.5 [19-28] | 37.9 [33-43] | 30.1 [25-35] | 27.9 [23-33] | 29.9 | 0.0% | 0.41 s | free tier |

Each decision model's row is its highest-mean setting from the grid, which is a post-hoc pick. The full
grid is in WORKLOG "2026-10-14: Part 2 wording x mode grid (hosted)" and in the run pages.

### Comparison with the paper

The paper evaluates only general LLMs on this benchmark (its Fig. 4), not decision models, so Part 2
has no paper values to compare against directly. The bridge is our Part 1 reproduction: the LLM rows
above come from `repro_fig4`, the same prompt and scoring as the paper's Fig. 4 setting, with thinking
at its minimum. The Part 1 write-up (`results/txt0_repro_fig4/writeup.md`) compares those LLM numbers
with the paper's.

### Key takeaways

1. **Decision models alone did not reach GPT-5's accuracy, but came close at a fraction of its cost and
   latency.** Jev (full wording, cumulative) reached 73.2% mean against GPT-5's 75.2%; on Q1 it was
   86.5% [82-90] against 93.4% [90-96]. It cost about 1/30 as much ($0.038 vs $1.24) and answered in
   0.70 s vs 3.29 s per item. On Q3 it scored 74.9% [70-79], above GPT-5's 64.3% [59-69].
2. **The best decision settings beat the small LLMs.** Jev (best setting, 73.2%) and OpenJev think 512
   (full wording, 72.7%) are above GPT-5 Mini (66.9%) and well above GPT-5 Nano (49.0%). At its default
   short wording, Jev (58.1%) still beat Nano (49.0%) at a fifth of Nano's cost ($0.014 vs $0.068).
3. **Setup choices moved accuracy more than the choice of system.** With the same model:
   - Jev: 58.1% (short wording, independent) -> 65.9% (full wording) -> 64.1% (cumulative) -> 73.2%
     (both).
   - OpenJev think 0: 37.1% (short) -> 64.7% (full wording); Q1 50.5% -> 78.4%, Q2 24.8% -> 67.1%.
   - luna Decisions: 51.5% (short, independent) -> 67.8% (full wording, cumulative).
   - Kev-4B: 45.7% (short) -> 55.6% (full wording).
   Full wording helped every system except Laya, whose full-wording Q3 input likely exceeded its
   512-token limit (see Caveats).
4. **Letting later questions see earlier answers (cumulative mode) helped some systems, not others.**
   Q2-Q4 rose for Jev (short wording: 52.7/49.5/48.0 -> 60.5/57.4/55.8) and luna Decisions
   (37.9/39.8/46.1 -> 51.1/50.2/53.9), but not for Kev-4B (38.2/58.6/42.9 -> 32.6/57.1/42.9). For
   OpenJev think 0 with full wording it was mixed: Q2 and Q4 fell, Q3 rose slightly
   (67.1/68.3/45.1 -> 45.1/69.9/41.4).
5. **Deciding vs generating, same model.** gpt-6-luna generating its answer (68.6%) beat luna's
   Decisions API at short wording (51.5%), but luna Decisions with full wording and cumulative mode
   (67.8%) came within a point of it, faster (0.52 s vs 2.66 s) and slightly cheaper ($0.062 vs
   $0.073). For Qwen3.5-4B the gap stayed: generating 64.1% (with 8.2% parse failures) vs reading its
   option probabilities 46.6%, and the two disagreed on many items (same answer on 190, 124, 94 and
   132 of 319 for Q1-Q4, against the 1024-cap generate run, txt27).
6. **Kev's decision training did not beat its own base model given one worked example.** Kev-4B scored
   45.7% (short wording) and 55.6% (full wording). Its base, Qwen3.5-4B-Base read through option
   probabilities, scored 44.7% zero-shot and 55.5% with one made-up worked example (txt29, the main
   base row by the pre-registered letter-probability rule).
7. **Option order and option names barely mattered for Jev.** Reversing the options changed Jev's
   per-question accuracy by at most 2.2 points (overlapping intervals). Letters instead of names raised
   Jev's Q1 from 82.1% to 86.2% and left Q2-Q4 within 2.2 points. For luna Decisions, letters lowered
   Q2 from 37.9% to 31.0%.
8. **Q1 (injury type) is where decision models are closest to the LLMs; Q4 is where they fall furthest
   behind.** Jev, luna Decisions and Qwen-Base one-shot all score above 80% on Q1 in their best
   settings, and every decision model's Q4 is below GPT-5's 65.5%.
9. **The open and local models have strong answer habits.** In the short-wording runs, "No realistic
   injury possible" (the answer key's choice on 5 items) was picked 208 times by Laya, 176 by Kev-4B
   and 131 by OpenJev think 0. Laya picked "moderate" on 306 of 319 Q4 items, and NLI picked "no
   injury" on 318 of 319 Q4 items.

### Caveats

- **Post-hoc selection.** The rows in the Summary table are each system's best of up to 8 settings,
  chosen after seeing all 319 items' results. A fair "best configuration" needs selection on training
  folds and scoring on held-out items, which is the Session 4 analysis. Treat the best-cell numbers as
  exploratory.
- **One run per setting, 319 items.** Per-question intervals are about +-5 points, so differences of a
  few points (e.g. option order, letters vs names) are within noise. Paired tests are not run here.
- **Different cost types.** Hosted APIs are priced per token; Codiv (OpenJev, Laya) was on a free tier;
  Kev-4B and the local rows ran on a rented Modal L40S GPU, with no per-call price. Latency for those
  rows is GPU compute time; network and queueing are excluded.
- **Refusals.** luna Decisions refused some questions (up to 2.5% of answers in one setting); they count
  as wrong.
- **Laya with full wording** (txt21): the Q3 input is about 750 tokens, over Laya's 512-token limit, so
  it was probably cut off. Unverified: Codiv reports no truncation.
- **Qwen3.5-4B generate** (txt33): 8.2% parse failures, mostly an invalid "E" on 4-option severity
  questions (29 answers) plus 11 replies cut off at the 2048-token cap. They count as wrong. This is
  above the 2% stop line set for text runs and is held for review.
- **Kev-4B** applies its own fitted temperature (2.41) to its probabilities: accuracy is unaffected, but
  its probabilities are not comparable with an untouched model's.
- **Qwen probability rows** use one prompt template (local wording, approved 2026-10-13); other
  templates could score differently. The two logprob rows were rerun after fixing a precision bug
  (bf16 ties); the numbers above are the fixed (fp32) runs.
- **Reference LLMs** are at thinking `min`; `repro_fig4_default` (thinking default) gives GPT-5 73.3%
  mean and GPT-5 Nano 66.1%.
- **Q2-Q4 refer to the Q1 injury.** A model whose Q1 differs from the key's can still be scored against
  the key's severity; strict accuracy is reported here. Q1-matched accuracies are in each run's
  `summary.json` (WORKLOG "Cumulative-mode scoring").
- **Timing.** These runs were made before `docs/HYPOTHESIS.md` was written.
