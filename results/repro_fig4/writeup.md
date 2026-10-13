### Summary

The large models reproduce the paper on risk-type recognition, but our severity scores are lower than the paper's, and the small-model gap looks different. GPT-5, Claude Opus 4.1 (4.5) and Gemini 2.5 Pro (3.1) all name the right injury type on more than 91% of the 319 scenarios, matching the paper's headline. Every model also does worse on the action questions than on the scenario-only questions, as the paper reports. Two things don't match: our severity accuracy is 5-29 points below the paper for 7 of 8 models, and GPT-5 Mini and Nano swap places (the paper has Mini well below Nano, we have the opposite).

### Comparison with the paper

**Risk recognition (Q1): reproduces.** The paper's claim is that the three large models all score above 90%, with an average of 92.3%. Ours are 93.4% (GPT-5), 91.8% (Opus) and 92.5% (Gemini Pro), an average of 92.6%. The paper's value sits inside our 95% CI for all three (e.g. Gemini Pro: paper 94%, ours 92.5% with CI 89.0-94.9%).

**Severity (Q2): lower than the paper.** On the exact 4-level answer, the paper's large models score 82-96% and ours 67-77%. The biggest gaps are the stand-ins: Gemini Pro (67.4% vs 96%), Sonnet (66.1% vs 92%) and Opus (73.7% vs 93%). One possible reading is that the paper's Fig 4a axis isn't the 4-level score. The paper's text describes a "highly severe or not" average of 88.7%. On that question, our three large models average 80.3%, and on recall of truly severe cases they average 93.9%, so the paper's number falls between our two versions of that metric. We kept the 4-level score as the axis (WORKLOG, severity decision), so this gap is partly a definition question rather than a model question.

**Small models lag (the paper's second claim): only partly.** GPT-5 beats Nano by 18-38 points on all four questions, all significant after Holm correction. But Mini is only 4.7 points behind GPT-5 on risk type, where the paper says 20. And the Gemini and Claude pairs show little or no gap: Opus beats Sonnet on 1 of 4 questions after correction, and Gemini Pro never beats Flash (it is 6.6 points lower on activated risk).

**Actions are harder (the paper's third claim): reproduces.** All 8 models score lower on action effect and activated risk than on risk type and severity, by 11-25 points, and none of the 95% CIs include zero. The paper's top scores on the action questions are 74% and 66%. Ours top out at 69.0% and 67.1% (both Gemini Flash).

### Key takeaways

1. **The paper's three headline claims hold in direction:** large models recognize latent risks well, small models can lag, and reasoning about an action's effect is harder than spotting the risk in a static scenario.
2. **Severity is where we diverge most,** and most of that is plausibly about what the paper measured (4-level vs severe-or-not), not model quality. Worth stating as a limitation and showing both versions (`scripts/plot.py --severity binary|recall`).
3. **The size gap depends on the family and on thinking.** GPT-5 Nano at minimal thinking is the clear outlier (55.5% on risk type). A separate run with default thinking (`repro_fig4_default`, txt1, still partial) brings Nano to about 87% on risk type, which is the paper's number. That suggests the paper may have used more reasoning than our `thinking: min`, and that "small models lag" is partly "small models at minimal thinking lag".
4. **The stand-ins limit the comparison.** Only the three GPT-5 models are the paper's exact models. The Claude 4.5 and Gemini 3.x stand-ins are newer models, so their gaps to the paper aren't evidence about the paper's models.
5. **The run is clean.** Parse failures are at most 0.6% for any model (2 Gemini safety blocks per Gemini model, 1 malformed GPT-5 Nano reply), with no truncated replies and no failed calls.

### Caveats

- All numbers are from one run at `thinking: min`, and each model answered each scenario once.
- The paper's Fig 4 values are read off its plots (five independent reads agree to 0.01), since the authors published no exact numbers.
- CIs are 95% Wilson intervals for single models. Gaps between models are paired over the same 319 scenarios (bootstrap CI, exact McNemar, Holm correction over the 16 size-gap tests).
