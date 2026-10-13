# Calibration, confidence-based cascades, and accuracy-vs-cost measurement

Scope: practical guidance for comparing one-forward-pass "decision models" (a probability per option) with
LLMs (GPT-5 family, Claude, Gemini) on a 319-item, 4-questions-per-item physical-safety MC benchmark, on
cost axes latency / tokens / $ / parameters, plus thinking budgets. Researched 2026-10-12. Items marked
**UNVERIFIED** come from background knowledge and were not confirmed against a fetched primary source in
this session.

---

## 1. Calibration metrics and pitfalls on small eval sets (Brier, ECE, reliability diagrams, bootstrap CIs)

### Takeaway
On ~319 items (or ~1,276 item-questions), the standard 15-bin equal-width ECE is noisy and biased, and the
bias is largest when a model is close to well calibrated. Report Brier score / NLL (proper scoring rules,
no binning) as the primary calibration numbers. Treat ECE as secondary: use equal-mass bins or a debiased
or sweep estimator, show a reliability diagram, and give bootstrap CIs resampled at the item level.

### Cited Findings
- Guo et al. define ECE as the bin-mass-weighted average of |accuracy - confidence| over M equal-width
  confidence bins, and use M = 15 for their reported numbers. — [Guo et al. 2017, On Calibration of Modern Neural Networks](https://ar5iv.labs.arxiv.org/html/1706.04599)
- HELM's example metrics include "ECE (10-bin)", so 10 bins is another common convention. — [Liang et al. 2022, HELM](https://ar5iv.labs.arxiv.org/html/2211.09110)
- The binned ECE estimator ("ECE_bin") can systematically under- or over-estimate the true calibration
  error, depending on the kind of miscalibration, the eval-set size and the number of bins. It is most
  biased for perfectly calibrated models. — [Roelofs et al. 2022 (AISTATS), Mitigating bias in calibration error estimation](https://arxiv.org/abs/2012.08668v3)
- Equal-mass (adaptive) bins give lower bias than equal-width bins. ECE_sweep, which picks the largest
  bin count that keeps the binned calibration curve monotonic, is less biased; so is a debiased estimator
  with equal-mass bins. — [Roelofs et al. 2022](https://research.google/pubs/mitigating-bias-in-calibration-error-estimation/)
- Kumar, Liang & Ma show that a debiased calibration-error estimator from meteorology needs O(√B) rather
  than O(B) samples. Their `uncertainty-calibration` Python package (pip) computes calibration error with
  bootstrap confidence intervals. They also report that Platt and temperature scaling are less calibrated
  than previously measured, because plug-in binned estimates under-measure the miscalibration of
  continuous outputs. — [Kumar et al. 2019 (NeurIPS), Verified Uncertainty Calibration](https://papers.nips.cc/paper/8635-verified-uncertainty-calibration); [code: p-lambda/verified_calibration](https://github.com/p-lambda/verified_calibration)
- **UNVERIFIED:** Nixon et al. 2019 ("Measuring Calibration in Deep Learning", arXiv 1904.01685)
  introduce adaptive-bin ECE and static/classwise calibration error (SCE). They argue that top-label ECE
  ignores the non-argmax probabilities, which matters in multi-class settings.
- **UNVERIFIED (standard result):** the Brier score is a proper scoring rule and decomposes (Murphy
  decomposition) into reliability - resolution + uncertainty. It needs no binning, which makes it more
  stable than ECE on small n.

### Inferences
- **Top-label vs multi-class.** For a decision model that outputs a full distribution over options, report
  (a) top-label ECE / reliability diagram on max-prob confidence and (b) multi-class Brier or NLL over all
  options. For LLMs that only give a letter, you only have top-label confidence, and only if you extract
  it (logprobs or verbalized). Not every compared system has the same calibration signal, so say which
  systems each calibration number covers.
- **Unit of resampling.** With 4 questions per item, questions within an item are correlated. Bootstrap
  over the 319 items, not the 1,276 questions (cluster bootstrap), or the CIs will be too narrow. Around
  1,000-2,000 resamples are cheap on a laptop.
- **Bin count.** With 319 points, 15 equal-width bins leave many bins with only a few points. 10 equal-mass
  bins (~32 points each), or ECE_sweep, is a defensible choice. Log the choice in WORKLOG (DECIDED /
  PROPOSED / OPEN) per the repo rules.
- **Parse failures.** Unparseable or refused LLM answers have no confidence. Either assign them
  confidence 1/k (uniform) or report calibration on the parsed subset with n shown. This is a scoring
  decision for the owner, not something to pick silently.

### Gaps
- I did not fetch Nixon et al. or a primary source for the Brier decomposition. Both are standard but
  marked UNVERIFIED.
- I found no source that gives an exact minimum n for a stable ECE. Roelofs et al. show bias vs n by
  simulation, but I did not extract their numeric curves.

---

## 2. Post-hoc calibration without fitting on eval items; verbalized confidence vs token probabilities; RLHF overconfidence

### Takeaway
Temperature scaling (one parameter, fit by NLL on held-out data) is the default. It cannot change accuracy,
so it is safe to report alongside raw accuracy. Platt and isotonic fits need more data than 319 items can
give. To avoid fitting on eval items, cross-fit (K-fold: fit T on K-1 folds, apply it to the held-out fold)
or fit on external data. RLHF/chat models' token probabilities tend to be overconfident. Verbalized
confidence is a black-box alternative that is sometimes better calibrated, but it is also overconfident.

### Cited Findings
- Temperature scaling is a single-parameter variant of Platt scaling. T is fit to minimize NLL on a
  held-out validation set with weights frozen. Because the argmax is unchanged, "temperature scaling does
  not affect the model's accuracy." — [Guo et al. 2017](https://ar5iv.labs.arxiv.org/html/1706.04599)
- In Guo et al., temperature scaling beat vector and matrix scaling (matrix scaling did poorly with many
  classes). Histogram binning helped but did not beat temperature scaling and can change predicted
  classes. Isotonic regression generally did not beat histogram binning. — [Guo et al. 2017](https://ar5iv.labs.arxiv.org/html/1706.04599)
- Kadavath et al.: large pretrained LMs are well calibrated on multiple-choice questions when shown
  "visible lettered answer options". Calibration improves with model size and with few-shot prompting. —
  [Kadavath et al. 2022, Language Models (Mostly) Know What They Know](https://ar5iv.labs.arxiv.org/html/2207.05221)
- Kadavath et al.: RLHF policies "naively appear very miscalibrated". A single temperature, "T=2.5 for all
  evaluations," largely fixes this across several tasks (§3.3, Fig. 9). — [Kadavath et al. 2022](https://ar5iv.labs.arxiv.org/html/2207.05221)
- Kadavath et al.: replacing an option with "none of the above" "reduces accuracy and calibration
  significantly", and models are biased against choosing it. — [Kadavath et al. 2022](https://ar5iv.labs.arxiv.org/html/2207.05221)
- Tian et al.: for RLHF models (ChatGPT, GPT-4, Claude), verbalized confidence is "typically
  better-calibrated" than conditional token probabilities on TriviaQA, SciQ and TruthfulQA. It often cuts
  ECE by a relative ~50%. — [Tian et al. 2023, Just Ask for Calibration](https://arxiv.org/abs/2305.14975)
- Xiong et al.: verbalized confidence "tend[s] to be overconfident". Calibration improves with model
  capability. Sampling-consistency and aggregation methods reduce overconfidence. White-box (logit)
  methods do better, but the gap is narrow (failure-prediction AUROC 0.522-0.605). No method dominates. —
  [Xiong et al. 2024 (ICLR), Can LLMs Express Their Uncertainty?](https://arxiv.org/abs/2306.13063)
- **UNVERIFIED:** the GPT-4 Technical Report (arXiv 2303.08774) includes an MMLU calibration plot. The
  pretrained base model is well calibrated, and the post-trained (RLHF) model is noticeably less so. A
  search confirmed the report exists but did not surface the passage.

### Inferences
- **Feasible protocol on 319 items with no eval leakage:** 5-fold cross-fitting at the *item* level. For
  each fold, fit one temperature (or Platt) on the other 4 folds and apply it to the held-out fold. Then
  report ECE/Brier on the concatenated out-of-fold predictions. This is a few lines of numpy/scipy and
  runs in seconds on an M1. It touches no labels from the item being scored, which fits the spirit of
  "never tune on eval items". Still, check with the owner/WORKLOG whether cross-fitting on eval labels is
  acceptable, or whether calibration must come only from external data.
- **Cleaner alternative:** fit T on an external physical-safety or commonsense MC set, if H1 training
  data already comes from elsewhere.
- Temperature scaling leaves accuracy unchanged. That keeps "calibrated vs uncalibrated" comparisons from
  muddying the accuracy axis, but it also means calibration alone cannot close an accuracy gap (H1).
- **Getting LLM confidences cheaply:** (a) logprobs of the option letter, where the API exposes them.
  Reasoning models often don't expose them, or the letter comes after hidden reasoning, so its logprob is
  near 1 and uninformative (**UNVERIFIED** generalization). (b) A verbalized confidence field. This
  changes the prompt, so per the repo rules it is a variant under `experiments/`, not the reproduction
  prompt. (c) Agreement across k samples (costs k×).

### Gaps
- I found no direct evidence on how calibrated GPT-5 / Claude 4.x / Gemini 2.5+ token probabilities are
  for MC answers after hidden reasoning, or whether logprobs are available when reasoning is on.
- Exact provider logprob availability in 2026 was not checked (OpenAI, Anthropic, Gemini).

---

## 3. Cascades and routing (FrugalGPT, LM cascades, AutoMix, RouteLLM, selective prediction); thresholds; reporting; savings

### Takeaway
A two-stage cascade works like this: the cheap model answers, and if its confidence is below threshold τ
the item goes to the expensive model. The literature reports large savings (≥2× up to "98%") at matched
quality. Those numbers come from text QA/chat benchmarks with learned routers, not from a 319-item safety
set. The honest way to present a cascade is a full accuracy-vs-cost (deferral) curve as τ sweeps 0→1,
plotted against the single models. If you report a single operating point, choose τ on held-out folds.

### Cited Findings
- FrugalGPT (Chen, Zaharia, Zou 2023) names three cost-reduction strategies (prompt adaptation, LLM
  approximation, LLM cascade). It learns which LLM combinations to use per query, and reports matching
  GPT-4 with "up to 98% cost reduction", or +4% accuracy at equal cost. — [FrugalGPT, arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- **UNVERIFIED (from memory of the full paper):** FrugalGPT's cascade uses a separately trained scorer
  (a small DistilBERT-style model) that rates each answer's reliability. Thresholds per stage are chosen
  by optimizing accuracy under a budget on a training split.
- Gupta et al. 2024: in classification cascades, the small model defers based on its predicted-class
  uncertainty. For generative LMs, naive sequence-level uncertainty suffers from length bias. Learned
  post-hoc deferral rules on token-level uncertainty do better, and embeddings from the small model (and
  intermediate layers of the large model) help further. — [Language Model Cascades: Token-level uncertainty and beyond, arXiv 2404.10136](https://arxiv.org/abs/2404.10136)
- AutoMix: the small model self-verifies with few-shot prompts. A POMDP-based router handles the noisy
  verification signal and picks a model size. It reports cost cut by "over 50% for comparable
  performance" across 5 LMs and 5 datasets. — [AutoMix (Aggarwal, Madaan et al., NeurIPS 2024), arXiv 2310.12963](https://arxiv.org/abs/2310.12963)
- RouteLLM: routers that pick between a strong and a weak LLM per query, trained on human preference data
  plus augmentation. Reports cost cut by "over 2 times in certain cases" without quality loss, and the
  routers transfer when the model pair is swapped. — [Ong et al. 2024, RouteLLM, arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
- Tian et al. present verbalized confidence as a way to defer low-confidence answers to a human expert,
  which is the same mechanism as a cascade. — [Tian et al. 2023](https://arxiv.org/abs/2305.14975)
- **UNVERIFIED:** Geifman & El-Yaniv 2017 ("Selective Classification for Deep Neural Networks") frame
  abstention as a risk-coverage trade-off: error on accepted items vs fraction accepted. They give a
  method to choose a threshold with a high-probability guarantee on risk. The area under the
  risk-coverage curve (AURC) is a standard summary.

### Inferences
- **Cascade from a decision model to an LLM in this project (all offline, ~free once both runs exist):**
  if you already have the decision model's per-option probabilities and the LLM's cached answers for all
  319 items, any cascade can be *simulated* without new API calls. For each τ: use the small model's
  answer where max-prob ≥ τ, otherwise the LLM's. Cost = small cost on all items + LLM cost on the
  deferred fraction. This fits the repo's "cache every response; reruns free" rule. Do it in a
  `scripts/` tool reading `results/`, not a notebook, if the numbers become results.
- **Per-question vs per-item deferral:** with 4 questions per item, deferring a whole item when any
  question has low confidence is the realistic cost model, because one LLM call answers all 4. Deferring
  per question overstates savings unless the LLM is called per question.
- **Threshold choice without overfitting:** (1) show the whole curve (main figure, no τ chosen); (2) if
  you must state one number, choose τ on K-1 folds to hit a target (e.g. "match the big model's
  accuracy" or "defer ≤ 30%"), evaluate on the held-out fold, and average across folds; (3) bootstrap
  CIs over items for the cascade's accuracy at that τ.
- **Upper bound to show:** an "oracle router" (defer exactly the items the small model gets wrong). It
  shows how much a perfectly calibrated confidence signal could save, and it is clearly not achievable.
- **Calibration and cascades interact:** temperature scaling changes neither the ranking of confidences
  nor, therefore, the deferral curve. Only τ's numeric value moves. Ranking quality (AUROC of confidence
  vs correctness) is what matters for cascades.
- Published savings (≥2×, >50%, up to 98%) come from settings where the small model is right on most
  queries. On a hard safety benchmark where the small model is weak, savings may be much smaller. Report
  your own curve rather than quoting these numbers as expectations.

### Gaps
- I did not fetch the full-text threshold-selection procedures for FrugalGPT, RouteLLM or AutoMix (only
  abstracts).
- I found no cascade paper specifically on safety/hazard classification or physical-safety judgments.
- Not checked: conformal "Learn then Test" style guarantees for choosing τ (Angelopoulos et al.). This
  may be overkill at n = 319.

---

## 4. Measuring cost: latency p50/p95, cold starts, local vs hosted comparability, $ per item, Pareto frontiers, pitfalls

### Takeaway
Report cost on several axes, each with its definition, and plot the accuracy-vs-cost Pareto frontier per
axis. Hosted-API wall-clock latency mixes in network, queueing, rate-limit backoff and provider load. It is
not comparable to local M1 latency or to scale-to-zero GPU latency (cold starts) unless you separate those
pieces. Tokens and $ per item are the most reproducible axes.

### Cited Findings
- HELM treats efficiency as one of 7 core metrics. It notes that efficiency "depends on both the model
  architecture and the hardware", and reports ~19,500 GPU-hours for open models and $38,001 of API spend
  for its evaluation. — [Liang et al. 2022, HELM](https://ar5iv.labs.arxiv.org/html/2211.09110)
- **UNVERIFIED:** HELM (§4.9) and a follow-up (Narayanan et al. 2023, "Cheaply Evaluating Inference
  Efficiency Metrics for Autoregressive Transformer APIs") distinguish raw API runtime, which is noisy
  because of shared hardware, batching and queueing, from "denoised" and "idealized" runtimes estimated
  on fixed hardware.
- Anthropic reports thinking tokens per request in `usage.output_tokens_details.thinking_tokens`. Thinking
  tokens are billed as output tokens. Changing the thinking budget between requests invalidates prompt
  cache breakpoints. — [Anthropic docs, Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- Anthropic recommends batch processing for thinking budgets above 32k tokens, to avoid timeouts. —
  [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- **UNVERIFIED:** Modal (and similar scale-to-zero serverless GPU hosts) adds container cold-start latency
  (seconds to tens of seconds for model weights) on the first request after idle. Batch APIs from
  OpenAI/Anthropic give ~50% discounts but no latency guarantee.

### Inferences (practical protocol)
- **$ per item** = (input_tokens × in_price + cached_input × cached_price + output_tokens incl.
  reasoning/thinking × out_price) / n_items. Use token counts from the provider's usage fields, which the
  repo already records via `models/reply_info.py`. Record the price table and the date it was read.
  Present $ per 1,000 items so tiny numbers stay readable.
- **Latency:** report p50 and p95 per model from per-request timings, measured *sequentially*
  (concurrency 1) on a separate warm run of e.g. 50 items. Throughput-oriented concurrent runs and
  rate-limit retries inflate tails. Exclude cache hits: cached replies have ~0 latency and must not enter
  latency stats. Store latency only from real calls.
- **Cold starts:** for Modal/scale-to-zero endpoints, report the first-request latency separately and
  exclude it (or send a warm-up call) from steady-state p50/p95. State the GPU type.
- **Local M1 vs hosted:** label local numbers with hardware (M1, 16 GB, backend such as MLX/llama.cpp/
  PyTorch-MPS, precision/quantization) and batch size 1. Do not put local and hosted latency on one axis
  without that label. Parameters and FLOPs/tokens are hardware-independent axes that sidestep this.
- **Decision models (one forward pass):** their output is 1 token-equivalent, so cost is dominated by the
  prompt (prefill). Count input tokens and use one forward pass as the latency unit. Avoid comparing
  "output tokens", which is ~0 for them by construction.
- **Pareto frontier:** for each axis, sort systems by cost and mark those no cheaper system beats on
  accuracy. Draw bootstrap CIs on accuracy as error bars; frontier membership within overlapping CIs is
  not a real difference at n = 319. Use a log scale on $ and latency.
- **Caching pitfall:** the repo caches responses, so any rerun's wall-clock is meaningless. Latency must
  come from the timing recorded at first call (it may be worth checking that the runner stores it).

### Gaps
- I did not fetch HELM §4.9 or the Narayanan et al. efficiency paper, so the denoised/idealized runtime
  details are UNVERIFIED.
- I did not fetch current Modal cold-start numbers or current API batch discounts.
- No current (2026) price tables were checked.

---

## 5. Thinking / reasoning budgets: accuracy vs reasoning tokens for classification/judgment; when extra thinking helps small models

### Takeaway
Provider docs say lower effort cuts latency and tokens, and many tasks are fine at low or medium effort.
Research shows longer reasoning can *hurt* on some judgment-like tasks (distractors, spurious features,
constraint tracking, AI-risk evals). Treat thinking budget as a measured axis (min vs default, 0 vs 512),
not an assumed gain.

### Cited Findings
- OpenAI GPT-5 guide: `reasoning.effort` ∈ {minimal, low, medium, high}, default medium. "minimal" is
  "our fastest option that still reaps the benefits of the reasoning model paradigm". Many workflows get
  consistent results at medium or even low. Start at medium and test the others on representative
  tasks. — [OpenAI GPT-5 guide](https://developers.openai.com/api/docs/guides/gpt-5)
- Newer OpenAI guides (GPT-5.5) list low/medium/high/xhigh (default medium), plus `none` for
  latency-critical use. A user-maintained forum table says supported values differ by model (e.g. `none`
  default on gpt-5.2). Check each model's page. — [OpenAI GPT-5.5 guide](https://developers.openai.com/api/docs/guides/gpt-5.5); [OpenAI community thread (unofficial)](https://community.openai.com/t/request-for-compatibility-matrix-reasoning-effort-sampling-parameters-across-gpt-5-series/1371738)
- Anthropic manual extended thinking: `budget_tokens` minimum 1,024, which must be < max_tokens. The
  budget is a target, not a cap. Returns diminish with budget depending on the task, at the cost of
  latency. For simple tasks, start near 1,024. `budget_tokens` is deprecated on 4.6 models and rejected
  on 4.7+, which use `thinking: {type: "adaptive"}` + `output_config.effort`. Under adaptive thinking at
  low effort, the model may skip thinking on easy inputs. — [Anthropic docs, Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- Gema et al. (Anthropic Fellows et al.), "Inverse Scaling in Test-Time Compute": longer reasoning
  *degraded* accuracy on simple counting with distractors, regression with spurious features, deduction
  with constraint tracking, and advanced AI-risk evals. Claude models got more distracted by irrelevant
  information. o-series models overfit to problem framings. The authors recommend evaluating across
  reasoning lengths. — [arXiv 2507.14417 (TMLR)](https://arxiv.org/abs/2507.14417)

### Inferences
- **Comparability of "min" across providers:** OpenAI "minimal"/"none", Anthropic adaptive low effort
  (or the 1,024 minimum on older models) and Gemini's thinking budget are not equal amounts of thinking.
  Report the *actual reasoning tokens used* per item (mean and p95) next to the setting name, and plot
  accuracy against real tokens.
- For OpenJev-style think 0 vs 512: a fixed small budget on a small model is a clean ablation. Report
  accuracy, Δtokens and Δlatency with paired bootstrap CIs (same items, both settings), since the
  difference is within-model.
- Physical-danger scenes may contain distractors (many objects, one hazard). The inverse-scaling findings
  suggest checking whether more thinking hurts on some categories. Report per-category deltas rather
  than assuming a gain.
- Thinking tokens also change calibration signals: with hidden reasoning before the answer letter, the
  letter's logprob tends toward certainty (**UNVERIFIED**, see §2).

### Gaps
- I found no source specifically measuring accuracy vs reasoning tokens for *small* models on
  classification or safety-judgment tasks. Whether extra thinking helps small models here is an open
  empirical question for the project.
- Gemini thinking-budget docs and OpenJev (razorback16) docs were not checked in this pass. See
  `docs/apis/`.
