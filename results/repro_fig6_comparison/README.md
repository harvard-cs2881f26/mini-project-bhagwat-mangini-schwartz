# Figure 6a: minimum versus provider-default settings

`fig6a.png`, `.pdf` and `.csv` cover all eight paper model slots, with successor models explicitly labeled. A percentage bar is shown only when all 164 responses are saved; incomplete conditions show response counts instead. Rates exclude malformed responses, and valid empty selections count as nonviolations. CSV/provenance retain the counts and source-summary hashes.

Claude Sonnet 4.5 and Opus 4.5 **send identical request parameters for min and default** in the shared client: neither label enables explicit thinking. Their two bars are independent stochastic samples, not distinct reasoning budgets. The separate 1024/4096-budget candidate experiments are not included here. Identical settings do not imply identical sampled outcomes.

Mini pairs `img2` min with `img3` default: both use system/user roles. Other pairs use user-only roles. The original user-only Mini min run remains preserved in img0 but is excluded from this matched-role comparison. These are within-model comparisons; prompt-role differences across model slots remain a limitation.

At this update OpenAI reports no credits remaining (Nano/GPT-5 defaults incomplete), Gemini Flash reports exhausted free-tier quota, and Pro has zero free-tier quota. Partial rates are deliberately not plotted as full-benchmark results. Flash-Lite stopped at 145/164 after 19 transient 503 failures; it can be resumed through the same runner. Caches prevent repeating saved requests. Exact author inference, parsing and aggregation settings remain unverified, and Figure 6b/c taxonomy plots remain provisional.

Regenerate after free CLI rescoring:

```bash
python scripts/plot_constraints.py configs/repro_fig6_default_remaining.yaml \
  --compare-thinking configs/repro_fig6_default.yaml configs/repro_fig6.yaml \
  configs/repro_fig6_successors.yaml configs/repro_fig6_mini_system.yaml \
  --output-dir results/repro_fig6_comparison
```
