# Decomposing the 4-question ASIMOV text task, and wording questions/options for decision models and option-logprob scoring

Researched 2026-10-12. Scope: ASIMOV-2.0 text benchmark (Q1 injury type, 5 options incl. "No realistic
injury possible"; Q2 severity; Q3 effect of instruction: Eliminates/Reduces/Same/Increases; Q4 severity
after the instruction). Q2-Q4 refer to "the injury you identified in Q1". Primary sources were read where
possible. Where only an abstract was read, the note says so. Anything not checked against a source is
marked UNVERIFIED.

## 1. Independent vs sequential/conditioned questions (chaining, joint vs separate, error propagation; API support)

### Takeaway
TypeSafe (Jev) and Kev both evaluate every question in isolation, so Q2-Q4 cannot see Q1's answer. OpenJev
has a `sequential` flag, but it conditions one *chunk* (~12 questions) on the earlier chunks' answers. It does
not condition question-by-question and is not documented in detail. TypeSafe's recommended pattern is to do the
chaining in your own code, e.g. a second call whose state includes Q1's answer. I found no study that directly
measures error propagation for chained versus independent hierarchical labelling. The main design question is
how to word Q2-Q4 so they stand alone.

### Cited Findings
- TypeSafe: "Every question is evaluated in parallel and in isolation against the same state in one go." Its advice is to "Ask each factor as a separate question, then combine the results with logic in your code." It also says to decompose a question that "would require extended reasoning or weighs multiple independent factors". And: "Each question is evaluated independently, so adding more questions does not create context-rot." — [TypeSafe docs intro](https://docs.typesafe.ai)
- TypeSafe patterns page lists "Speculative Fan-Out" ("Send many questions in a single call, including speculative ones, and let your code decide what's relevant"), "Confidence-Gated Routing", "Composite Scoring" and "Intent Routing". There is no dedicated chaining primitive. — [TypeSafe patterns](https://docs.typesafe.ai/patterns)
- OpenJev `sequential`: "For long question lists: read the chunks in order. Each chunk sees the answers before it." By default the server reads questions in chunks of about 12 and runs the chunks in parallel. `sequential` costs one read per chunk, in series. "`think` and `sequential` need a text state". With images the request gets a 400. The text-only backends (Laya, Verdict, CLM, JevK5) also return 400 for `sequential`. The README does not say how earlier answers are shown to the model. — [OpenJev README](https://github.com/razorback16/openjev)
  - Inference from the chunk size: our 4 questions fit in one ~12-question chunk, so `sequential` would probably *not* make Q2 see Q1. UNVERIFIED; test with a `--limit` run, or split Q1 into its own request.
- Kev: questions run with exact isolation. "each question runs as its own row, so 'isolation is exact'". Packed and separate runs agree to within 4e-6 in fp32 tests. `/v1/systemone/separate` runs each question in its own forward pass. Kev has no sequential option. — [Kev README](https://github.com/jaredpalmer/kev); also summarised in repo `docs/apis/systemone.md`
- OpenAI Decisions API: per repo notes, the `choice` type takes `choices: [{value, description}]` and `score` takes `levels: [{label, description}]`. I found no conditioning/sequential option. — repo `docs/apis/openai_decisions.md` (sources: [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)); third-party summaries agree on fields: [promptfoo](https://www.promptfoo.dev/docs/providers/openai-decisions/). Whether Decisions questions in one call are isolated is UNVERIFIED.
- Annotation literature: one paper on order sensitivity in LLM annotation found that label order and question wording can flip outputs. It proposes querying each label separately and normalising the probabilities to remove positional bias. — [arXiv 2502.19679](https://arxiv.org/html/2502.19679v4) (read via search snippet only)
- Chain designs where each stage sees the previous stage's labels report accuracy gains. They also give an early error a direct path into later stages. — [arXiv 2409.09615](https://arxiv.org/html/2409.09615v1) (search snippet only)
- Error-decomposition study: prompting strategies reduce model-specific errors but "cannot resolve the ambiguity embedded in the task formulation". — [arXiv 2601.11920](https://arxiv.org/pdf/2601.11920) (search snippet only)
- Least-to-most / decomposed prompting (Zhou et al. 2022, arXiv 2205.10625; Khot et al. 2022, arXiv 2210.02406) feed earlier sub-answers into later prompts. They target multi-step *reasoning* with generation, not one-pass classifiers. UNVERIFIED in this session (not re-read).

### Inferences
- Three feasible designs for decision APIs:
  (a) **Independent**: one call with 4 questions; Q2-Q4 reworded so they don't depend on Q1, e.g. "the most
  likely injury in this scenario". Cheapest. Q2/Q4 may pick a different injury than Q1.
  (b) **Two-stage chain in our code**: call 1 = Q1; call 2 = Q2-Q4 with the state extended by "Most likely
  injury: <Q1 answer>". Costs 2 calls, adds latency, and propagates Q1 errors (a wrong Q1 injury conditions
  Q2-Q4). Comparing (a) and (b) is directly measurable on the 319 items.
  (c) **Marginalised chain**: run call 2 once per Q1 option and weight the results by Q1's probabilities. Costs
  about 5x for the second stage. This is my own idea, not taken from any source.
- The paper's answer key presumably scores Q2-Q4 against the key's own injury (check the data/answer-key
  definition). If so, design (a) asks a question closer to "severity of the most plausible injury", which may
  match the key's intent. This is an open scoring question: per CLAUDE.md, stop and ask the owner.
- Q3 ("effect of the instruction") and Q4 depend on the instruction text, not strongly on Q1. They are likely
  the least harmed by independence.

### Gaps
- No direct empirical comparison found of chained vs independent multi-question classification for one-pass
  decision models.
- OpenJev does not document how earlier chunk answers are shown under `sequential`, and the exact chunk size
  is approximate ("about 12").
- I could not confirm whether OpenAI Decisions isolates questions within a call.

## 2. Instruction length: full rubric/few-shot vs short instructions

### Takeaway
Vendor guidance for decision APIs favours short "gut-check" questions with a one-line description per option.
It adds structure only when options get confused. I found no controlled study on rubric length for small or
encoder classifiers. Some backends have hard token budgets: Laya's options share 256 tokens, so a long rubric
gets cut. Prompt-format sensitivity in small models is large and does not go away with scale or instruction
tuning.

### Cited Findings
- TypeSafe: "Think of each question as a gut-check determination". `instructions` and each `criteria` entry
  can be a string, object or array. "Start with a string." "Start with a one-line description per option."
  "When two options are similar and the model keeps confusing them, describe each one with an object instead
  of a string." The object field names (`question`, `focus`, `what`, `not_for`, `examples`) "are not part of
  the API". — [TypeSafe Choice](https://docs.typesafe.ai/primitives/choice); [TypeSafe intro](https://docs.typesafe.ai)
- Laya (OpenJev family): "Its options share 256 tokens, so with many options each option is cut to a few
  tokens. Use about 20 at most, or split the question." Laya windows are 512/1,024 tokens. JevK5 allows 16,384
  tokens. — [OpenJev README](https://github.com/razorback16/openjev)
- Kev limits: states up to 65,536 tokens plus 8,192 per question. There is no guidance on instruction length.
  — [Kev README](https://github.com/jaredpalmer/kev)
- Sclar et al.: meaning-preserving formatting changes cause "performance differences of up to 76 accuracy
  points" (LLaMA-2-13B, few-shot). "Sensitivity remains even when increasing model size, the number of few-shot
  examples, or performing instruction tuning." Format performance "only weakly correlates between models". The
  authors recommend reporting a range across plausible formats (FormatSpread). — [arXiv 2310.11324](https://arxiv.org/abs/2310.11324) (abstract)
- Lost in the middle: retrieval/QA accuracy is highest when the relevant information is at the start or end
  of a long context, and it degrades in the middle. These were retrieval tasks, not classification rubrics. —
  [Liu et al., TACL 2024](https://aclanthology.org/2024.tacl-1.9/)

### Inferences
- Q3's worked examples and Q2's severity rubric are the long parts. For one-pass classifiers, a natural option
  is to move the rubric text *into each option's description*, e.g. "Moderate: requires medical treatment but
  not life-threatening…", and keep `instructions` short. This follows the TypeSafe guidance.
- Paper-verbatim prompts stay for reproduction (CLAUDE.md hard rule). Short/rubric-in-options wordings are
  variants and belong in `experiments/` under their own run name. Given Sclar et al., testing 2-3 wordings and
  reporting the spread is better evidence than a single wording.
- Check the token budget per backend before sending the full rubric. Laya would silently truncate options.

### Gaps
- No controlled evidence found on rubric/few-shot length vs accuracy for small (<10B) or encoder/decision
  models in classification. Rubric-anchored severity scoring for safety specifically was not found.

## 3. Option naming: letters vs semantic names vs descriptions

### Takeaway
Option-ID letters cause much of the selection bias in first-token scoring: models carry a prior over "A/B/C/D".
Decision APIs send both the option *name* and its description to the model. Semantic names with
distinguishing descriptions are what the vendors recommend. Letter names also re-introduce letter-token
priors on letter-scoring backends.

### Cited Findings
- PriDe (Zheng et al., ICLR 2024): selection bias "primarily stems from LLMs' token bias", meaning a prior
  probability mass on ID tokens. Removing option IDs (likelihood scoring of content instead) sharply cuts bias.
  gpt-3.5-turbo RStd went 5.5 -> 1.0 (MMLU) and 3.3 -> 0.6 (ARC). Averaged over models, RStd fell by
  6.4/5.1/6.2, but accuracy dropped by 2.1/2.9/7.0 points (MMLU/ARC/CSQA). — [arXiv 2309.03882 (HTML)](https://arxiv.org/html/2309.03882)
- Robinson et al. (ICLR 2023): showing all options and asking for the symbol (MCP) beats cloze scoring for
  models with high "multiple choice symbol binding" (MCSB) ability, which "varies greatly by model". —
  [arXiv 2210.12353](https://arxiv.org/abs/2210.12353) (abstract)
- TypeSafe Choice: "The option names and their descriptions are both sent to the model, so write descriptions
  that separate the options from each other." Use `null` descriptions "because the option names are clear on
  their own". "Add an `other` or `none of the above` option when the list might not cover every input." —
  [TypeSafe Choice](https://docs.typesafe.ai/primitives/choice)
- OpenJev (DiffusionGemma backend): "Each label is one token: `yes`/`no` for a `noul`, `A`/`B`/`C` for a
  choice, `0`/`1`/`2` for a score." So the server itself letters the options internally, and the user's names
  appear as text. JevK5's answer is a softmax over letter logits. Question ids are never sent ("It sees `q1`,
  `q2`, `q3`"). — [OpenJev README](https://github.com/razorback16/openjev)
- Kev scores each option's `</opt>` hidden state against the question's `<decide>` state with a pointer head,
  so it scores option *content positions* rather than letter tokens. — [Kev README](https://github.com/jaredpalmer/kev)
- Current repo plan: OpenAI Decisions `value` = letters "A".."E" with option text as `description`; Jev
  `criteria` = `{"A": "<option text>", ...}`. — repo `docs/apis/openai_decisions.md`, `docs/apis/systemone.md`

### Inferences
- With letter names, the model gets the letter twice (our "A" plus the server's internal letter on
  letter-based backends). Semantic names like `no_injury`, `minor`, `moderate` carry meaning on their own.
  Given PriDe's token-bias finding, semantic names are worth a variant run. Mapping back to the answer key's
  letters is trivial in code.
- For the small-LLM logprob path: scoring the letter after a listed-options prompt (MCP) relies on the model's
  symbol-binding ability, which is weak in small/base models per Robinson et al. Scoring each option's text
  likelihood (cloze) avoids letter priors but costs accuracy on average (PriDe numbers above) and needs length
  normalisation.
- Q1's "No realistic injury possible" works as the documented "none" fallback. Keep it, and give it a
  distinguishing description.

### Gaps
- No head-to-head study found of letter vs semantic option names *in decision APIs* specifically.

## 4. Option-order (selection) bias: size, debiasing by permutation, cost, Kev /permute

### Takeaway
Order effects are large in LLM MCQ evaluation: up to a 13-75% accuracy gap from reordering (Pezeshkpour &
Hruschka). Cyclic permutation (n passes for n options) gives about +5-8 accuracy points on average. PriDe gets
most of the bias reduction for about 1.15-1.2x cost. Jev is documented to lean toward the first option. Kev's
`/permute` runs one Choice question over 1-64 orders (default 6), but its README does not say how orders are
chosen or how results are aggregated.

### Cited Findings
- Pezeshkpour & Hruschka: reordering options gives "a considerable performance gap of approximately 13% to 75%"
  across benchmarks, and the gap persists with few-shot demos. They conjecture the sensitivity arises when the
  model is uncertain between its top 2-3 choices. Putting the top two as the first and last options amplifies
  the bias. Two calibration methods give "up to 8 percentage points improvement". — [arXiv 2308.11483](https://arxiv.org/abs/2308.11483) (abstract)
- Zheng et al. (PriDe):
  - Cyclic permutation pairs every option with every ID. It costs n passes (x4 for 4 options, x5 for 5).
    Average 0-shot gains are +4.9 (MMLU), +5.7 (ARC), +7.9 (CSQA) accuracy, and RStd falls by 8.7/8.6/11.2.
    Full permutation needs n! passes (24 for 4 options), which the authors call prohibitively expensive.
  - PriDe estimates the ID prior from 5% of test samples, with no labels needed. Cost is x1.15 (MMLU/ARC) or
    x1.2 (CSQA). It gives accuracy +1.2/+1.3/+1.7 and RStd -7.6/-5.6/-6.9. Using 80% of samples (x3.4 cost)
    gives +4.1/+4.9/+6.6.
  - Neither size nor instruction tuning reliably reduces the bias. vicuna-33B (RStd 8.2) is more biased than
    vicuna-7B (5.3), and llama-2-70B (4.1) is less biased than llama-2-chat-70B (12.8).
  - Preferences vary by model: llama-30B picks A 34.6% vs D 15.8% on balanced data; gpt-3.5-turbo favours C.
    Moving the gold answer to D cut gpt-3.5-turbo's accuracy 67.2 -> 60.9.
  — [arXiv 2309.03882 (HTML)](https://arxiv.org/html/2309.03882)
- Kev `/v1/systemone/permute`: "one Choice question with different option orders (`n_perm` 1 to 64, default
  6)". It applies to Choice only; Score questions are not permuted. The README does not specify random vs
  cyclic orders, the aggregation rule, or cost. Kev also says "options within a question can still affect one
  another" and "Changing option order can change an answer." — [Kev README](https://github.com/jaredpalmer/kev)
- Jev's docs list "Choice option order" as a known weakness: jev-1.13 leans toward the **first** option. —
  repo `docs/apis/systemone.md` (from TypeSafe docs; the original page was not re-read this session, so
  UNVERIFIED here)
- OpenJev re-reads uncertain slots (entropy > 0.1) three more times with fresh noise and averages the four
  results. That reduces sampling noise, not order bias. — [OpenJev README](https://github.com/razorback16/openjev)

### Inferences
- Q1 has 5 options, so cyclic permutation costs 5x; Q2-Q4 have 4 options, so 4x. At Jev's price ($0.042/M
  input) and the repo's ~$0.01-0.03 per full pass estimate (UNVERIFIED), even 5x stays cheap in dollars. The
  cost to report is latency and token count, which is relevant to H1/H2's cost axis.
- For the ordinal questions (Q2, Q4) and Q3's ordered scale, permuting breaks the natural order the model may
  rely on. Cyclic shifts keep the relative order but wrap it around. A full reversal is another cheap check. It
  is better to report "bias size" (accuracy/answer distribution under 2+ orders) than to always debias.
- Kev's 6 default orders cannot be a full cyclic set for both 4 and 5 options. To know whether a result is
  "cyclic-debiased", read Kev's source (`kev_serve.py`) for the ordering and aggregation rule before relying on
  it.

### Gaps
- Kev `/permute`'s permutation scheme and aggregation (mean of probabilities? majority?) are undocumented in
  the README. The source code was not read.
- No published order-bias measurements were found for Jev/OpenJev/Kev/OpenAI Decisions on any benchmark.

## 5. Logprob (first-token/option-likelihood) scoring vs generated answers

### Takeaway
For instruction-tuned chat models, first-token option probabilities often disagree with the model's text
answer: over 60% mismatch in Wang et al. 2024a. When they disagree, the text answer is more robust to
perturbation and to option order, even compared with PriDe-debiased first-token scores (Wang et al. 2024b).
Option-logit scoring is the native interface for decision APIs and fits base models better. For small
*instruct* LLMs, it should be checked against generated answers on a sample.

### Cited Findings
- Wang et al. 2024a ("My Answer is C"): first-token probabilities and text outputs are "severely misaligned on
  all dimensions, reaching mismatch rates over 60%". The dimensions are final choice, refusal rate, choice
  distribution and robustness. Models "heavily fine-tuned on conversational or safety data are especially
  impacted". They "remain misaligned even when we increasingly constrain prompts". Causes include responses
  that start with "Sure" or with a refusal. — [arXiv 2402.14499](https://arxiv.org/abs/2402.14499) (abstract; ACL 2024 Findings)
- Wang et al. 2024b ("Look at the Text"): text answers are more robust to question perturbations than first
  token probabilities when the two mismatch. When mismatch exceeds 50%, the text answer is more robust to
  option-order changes than first-token probabilities debiased with PriDe. — [arXiv 2404.08382](https://arxiv.org/abs/2404.08382) (via search snippet of abstract)
- PriDe: likelihood scoring of option content (no IDs) lowers selection bias but usually costs accuracy, and
  the authors call it less convenient. — [arXiv 2309.03882](https://arxiv.org/html/2309.03882)
- OpenJev scores only the label tokens, so "an answer cannot go off-schema". With `OPENJEV_CONSTRAINED=1`,
  "Probabilities are normalized over the labels". Confidence = 1 - normalised entropy. — [OpenJev README](https://github.com/razorback16/openjev)
- Kev confidence is `(p_max - 1/K)/(1 - 1/K)`. Probabilities are relative to the supplied options only. —
  repo `docs/apis/systemone.md`; [Kev README](https://github.com/jaredpalmer/kev)
- Overconfidence: one study notes the LLM can be "spuriously certain" and put near-100% on the chosen answer.
  Its fix was raising temperature. — [arXiv 2406.10267](https://arxiv.org/pdf/2406.10267) (search snippet)
- A 2026 calibration paper treats frozen readers' option logits as noisy ordinal measurements, because the top
  probability is often saturated near 1. It fits a small channel from a few labels. — [CORDIAL, arXiv 2609.29807](https://arxiv.org/pdf/2609.29807) (search snippet only; UNVERIFIED details)

### Inferences
- For a small instruct LLM (e.g. Qwen3-4B) scored via option logits: on a `--limit` sample, also generate the
  text answer and report the agreement rate. Wang et al. show this can be low for chat/safety-tuned models,
  and safety-tuned behaviour is exactly what is relevant in a physical-danger benchmark.
- Tokenisation: check that each option label is a single token, with or without a leading space ("A" vs " A"),
  in the small model's tokenizer. Otherwise first-token scoring compares unequal things. This is standard
  practice, not taken from a source read here (UNVERIFIED).
- Refusals: logit scoring cannot "refuse", so a model that would refuse in generation gets a forced answer.
  This interacts with the repo's `refusal_rate` convention, which logit-scored runs cannot fill. The run should
  record that it was not applicable rather than 0.

### Gaps
- No base-vs-instruct agreement numbers were read in full text this session.
- Per-model mismatch rates (Wang et al.) were not read; only the abstract headline (>60%).

## 6. Ordinal labels (severity): ordinal vs nominal treatment; metrics

### Takeaway
Decision APIs already have an ordinal primitive. `score` returns the expected level Σ i·p_i (0-indexed) plus
the full distribution. Probability-weighted (expected-value) scoring is established for LLM graders (G-Eval)
and gives small correlation gains over the argmax. For accuracy against a categorical answer key, the argmax
is still needed. MAE or weighted kappa on the level index would show how far wrong a miss is.

### Cited Findings
- OpenJev `score`: returns `{score: Σ i·pᵢ, legend, probabilities, confidence}` (expected level, 0-indexed).
  Caution: "Evaluate score questions on your own data before you rely on them." The CLM backend's score
  questions can ignore the state. — [OpenJev README](https://github.com/razorback16/openjev)
- Kev `score`: criteria are "1–255 descriptions, ordered from lowest to highest". The answer is the mean level
  index from 0, plus `legend`/`probabilities`/`confidence`. Score questions are not permutable. — [Kev README](https://github.com/jaredpalmer/kev)
- TypeSafe `score`: `criteria` = ordered array of 2-10 levels. OpenAI Decisions `score`: `levels: [{label,
  description}]` from lowest to highest. On OpenAI, "the model reads each level as `label: description`". —
  repo `docs/apis/systemone.md`, `docs/apis/openai_decisions.md`; [OpenJev README](https://github.com/razorback16/openjev)
- G-Eval uses the probabilities of the predefined score tokens and takes their weighted sum as the final score,
  for "more fine-grained, continuous scores". Ablation on SummEval: Spearman 0.514 with probabilities vs 0.502
  without. Kendall-τ was 0.418 vs 0.446, which the authors attribute to ties in integer scores. — [arXiv 2303.16634 (HTML)](https://arxiv.org/html/2303.16634)
- Expected value over answer-letter probabilities, E = Σ k·p(k), was used to improve reading comprehension
  scoring. — [arXiv 2406.10267](https://arxiv.org/pdf/2406.10267) (search snippet)

### Inferences
- Q2/Q4 (No/Minor/Moderate/Severe) map naturally to a `score` question with 4 levels, with rubric text as each
  level's description. The categorical answer is argmax(probabilities), and Σ i·p_i is available as a
  secondary ordinal signal. This is a design option for the owner. Whether to use `choice` or `score` changes
  the request and possibly the model's behaviour, so it is a variant run.
- Q3 (Eliminates/Reduces/Same/Increases) is also ordered (risk change from most reduced to increased). It can
  be treated as ordinal too, but "Eliminates" vs "Reduces" may not be equally spaced.
- Metrics alongside the paper's accuracy: MAE on the level index, quadratic-weighted Cohen's kappa, and the
  off-by-one rate. These separate "Moderate vs Severe" slips from "No vs Severe" errors. Adding any of them is
  a scoring decision to log in WORKLOG (owner decides).

### Gaps
- No source found evaluating `score`-type (expected-level) outputs of decision APIs against ground truth.
- No study found on weighted kappa/MAE specifically for LLM safety-severity labels. These metrics are standard
  in ordinal classification, but no citation was retrieved here (UNVERIFIED as a citation).
