# System One / decision-model architectures: accuracy, cost, when to use (read 2026-10-12)

Scope: models that answer a question with a probability per option from one (or a few) forward passes instead of generated text, as candidates for ASIMOV-2.0 **text** (319 items; per item a scenario + robot instruction and 4 multiple-choice questions: injury type (5 options), latent severity (4), effect of the instruction on injury likelihood (4), severity after the instruction (4)). Our machine: M1 Pro, 16 GB. All numbers below are the vendors'/authors' own unless stated; none were reproduced. Related repo notes with interface details: `docs/apis/systemone.md`, `openai_decisions.md`, `semif.md`, `nli_zeroshot.md`, `qwen3_4b_local.md`.

## Q1. Architecture families and how each produces option probabilities

### Takeaway
There are five mechanisms: (a) trained decision heads that score option-marker tokens (Kev's pointer head, Laya's [MASK] scorer, Verdict's GLiClass head, Clef's all-options head); (b) reading label-token probabilities from a frozen generative or diffusion LM (SemIf, OpenJev, option-letter logprobs); (c) NLI entailment softmax (zeroshot-v2.0); (d) closed hosted services whose mechanism isn't documented (Jev, gpt-6-luna Decisions); and (e) safety classifiers that output P(unsafe) or P("Yes") for one policy. Families (a)-(d) natively return a distribution over N runtime-defined options. Safety classifiers are binary per policy, so a 4- or 5-way severity or effect question has to be rebuilt as one yes/no policy per option. They aren't built for that.

### Cited Findings
**Hosted closed services**
- TypeSafe Jev (`jev-1.13.0`, aliases `jev-latest` / `jev-preview`). It returns typed answers (Choice, Score, Noul) with `probabilities` and `confidence` and doesn't generate text. It is "trained for calibrated decisions", and the docs don't say how the probabilities are produced. — [TypeSafe docs](https://docs.typesafe.ai); [System One concept](https://docs.typesafe.ai/concepts/system-one)
- Jev is trained with RLCD, isn't fine-tuned on customer data, and takes text input only. — [TypeSafe Models](https://docs.typesafe.ai/models)
- Questions in one call are "evaluated in parallel and in isolation against the same state". Adding questions "barely changes response time". — [TypeSafe docs](https://docs.typesafe.ai)
- OpenAI Decisions API (public beta since 2026-10-06; `gpt-6-luna` is the only model) has `predicate`, `choice` and `score` question types. `choice` returns `probabilities` over the options plus a `confidence` field the docs don't define. `score` is the probability-weighted level index. Questions are independent. — [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)

**Trained decision heads on encoders / small LMs**
- Kev (0.8B / 4B / 9B): a Qwen3.5 base + rank-16 LoRA + a "pointer head". It scores each option's `</opt>` hidden state against the question's `<decide>` state, then applies a softmax and a per-checkpoint fitted temperature (the temperature doesn't change the argmax). Kev-27B is a full fine-tune of Qwen3.8-27B. Questions are isolated (an attention mask, or one row per question over a cached state for DeltaNet layers). — [Kev README](https://github.com/jaredpalmer/kev)
- Laya 1.0: a ModernBERT-large backbone (395M) + a decision head (2 transformer layers, an option-marker scorer and an act/escalate head), 421M in total. Each option is scored at its own [MASK] token, with a softmax across the question's options. Trained with RLCD using strictly proper scoring rules. There is also a multilingual mmBERT-base variant (322M). — [Laya model card](https://huggingface.co/convaiinnovations/laya)
- Verdict 1.4: a ModernBERT encoder with a GLiClass head, "calibrated per option count", 151M, up to 24 choices, about 7 ms per request. — [Codiv](https://codiv.ai)
- CLM 0.1 (Contrastive-LM): "contrastive state and action heads over Qwen3-8B embeddings", 2,048-token context, about 100 ms. — [Codiv](https://codiv.ai)
- JevK5 0.3: Qwen3.5-4B with a LoRA "distilled from a thinking teacher", 16,384-token context, about 80 ms per question. — [Codiv](https://codiv.ai)
- Clef-flash 1.0 (Cloudflare): Qwen3.5-9B with a head that "scores every option of every question in one pass" (NVFP4), text and images, 16,384 tokens, about 30 ms. — [Codiv](https://codiv.ai)

**Label-token readout from a generative / diffusion LM**
- OpenJev 0.1 (razorback16; served on Codiv as `openjev-latest`). It uses DiffusionGemma 26B-A4B (a discrete diffusion LM, 4B active), masks one answer-slot token per question, and runs one read-only forward pass that scores only the label tokens, so answers can't go off-schema. If any slot's entropy is above 0.1, it re-reads three more times with fresh noise and averages; those re-reads aren't billed. Confidence = 1 − H(p)/ln K. Optional `think` (0-4096 tokens) writes a thought first. Input is then billed twice and the thought counts as output. The README advises 512+ for multi-step problems but reports no accuracy measurement for `think`. — [OpenJev README](https://github.com/razorback16/openjev)
- SemIf (formerly OpenJev; TheoLeeCJ, MIT): a frozen Qwen3.5-4B. One forward pass reads the logits of the declared option tokens and applies a softmax; no decoding. Probabilities are "conditional on the supplied options". A shared state can be prefilled once and the criteria scored as branches. — [SemIf README](https://github.com/TheoLeeCJ/SemIf)
- Plain option-letter logprob scoring (for example Qwen3-4B, with a softmax over the A-E letter logits): first-token probabilities can disagree badly with the model's own text answer. The paper reports mismatch rates over 60%, worst for models heavily tuned on conversation and safety data, and the gap persists even when prompts force an option-letter start. — ["My Answer is C", arXiv 2402.14499 (Findings ACL 2024)](https://arxiv.org/abs/2402.14499v1)
- Follow-up: text answers are more robust to question perturbations than first-token probabilities when the two disagree. Above 50% mismatch they also beat PriDe-debiased first-token probabilities under option reordering. — ["Look at the Text", arXiv 2404.08382](https://arxiv.org/abs/2404.08382)

**NLI zero-shot**
- `zero-shot-classification` turns each (text, label) pair into a premise and hypothesis. With `multi_label=False` it applies a softmax over the entailment logits across labels. Cost is one forward pass per option. DeBERTa-v3-large-zeroshot-v2.0 has a 512-token limit; ModernBERT-large-zeroshot-v2.0 has 8,192. — [HF pipeline docs](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline); [deberta-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0); [ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0)

**Cross-encoder / reranker style**
- SemIf's "native reranker" mode (scoring each option as a query-document pair) did worse than direct option logits on its general baseline. It scored 0.625 vs 0.813 on 144 authored decisions, 0.522 vs 0.637 on WANLI and 0.560 vs 0.845 on the TypeSafe subset, and tied only on retrieval. — [SemIf README](https://github.com/TheoLeeCJ/SemIf)

**LLM safety classifiers**
- Llama Guard generates text ("safe" / "unsafe" plus the violated categories). Its classifier score is the probability of the first output token, used as P(unsafe). Llama Guard 4 is a 12B early-fusion multimodal model that uses the MLCommons hazard categories (14 in Llama Guard 3). The first-token scoring method is documented for LG3; for LG4 it is **UNVERIFIED**. — [Llama Guard 4 docs](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama-guard-4/); [Llama Guard 3 8B card (Azure)](https://ai.azure.com/catalog/models/Llama-Guard-3-8B)
- ShieldGemma (2B / 9B / 27B, Gemma 2): a softmax over the "Yes" / "No" next-token logits, where P(Yes) is the violation score. The user supplies the policy text. The card gives ready-made policies for four harm types. "Performance is highly sensitive to how the safety principle is worded." — [ShieldGemma card](https://huggingface.co/google/shieldgemma-2b)
- Granite Guardian 3.3 8B returns a yes/no score in `<score>` tags, with optional `<think>` reasoning. It accepts a user-written risk definition via `custom_criteria` ("custom criteria require testing"). English only. Apache-2.0, released 2025-08-01. — [Granite Guardian 3.3 8B card](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b)
- gpt-oss-safeguard-20b (21B total, 3.6B active; Apache-2.0): policy-as-prompt with reasoning (effort low / medium / high), harmony format only, fits on 16 GB of VRAM. The card doesn't say it returns probabilities. Intended for input/output filtering and trust-and-safety labeling. — [gpt-oss-safeguard-20b card](https://huggingface.co/openai/gpt-oss-safeguard-20b)

### Inferences
- Every safety classifier above answers one binary question: does the content violate this policy? Mapping a 4-option severity or effect question onto one would take a policy per option (4-5 calls, each giving an independent P(Yes)) and a renormalisation that these models weren't trained for. Granite's `custom_criteria` and ShieldGemma's free-text policy make that possible, but nothing published shows them working on ordinal multi-class questions. Their harm taxonomies are about content harm, not physical injury severity. For H2, they fit Q3 best if Q3 is collapsed to a binary question ("does the instruction increase injury risk?"), and a 4-way severity question worst. The owner decides.
- For ASIMOV, all non-safety families can take the 4/5 options as runtime criteria with no training, and none requires parsing JSON.

### Gaps
- TypeSafe and OpenAI don't say how Jev or gpt-6-luna turn the model into probabilities. Whether Decisions `confidence` is max-p, 1−entropy or something else is undocumented.
- I didn't read Llama Guard 4's card in full for its scoring method or custom-category support. ShieldGemma's context length isn't on its card.
- I found no primary card for Verdict, CLM, JevK5 or Clef beyond Codiv's one-line descriptions.

## Q2. Published accuracy / calibration numbers and known weaknesses

### Takeaway
Most published numbers come from the authors, on routing and triage style data. The reported ranking is roughly hosted Jev ≥ Kev-27B > Kev-4B/9B ≈ SemIf-4B > base encoders. Laya and Verdict-class encoders need fine-tuning to beat simple baselines. TypeSafe publishes **no** accuracy, calibration or latency figures for Jev itself; Jev numbers appear only in third-party READMEs. Nothing has been published on physical-safety or counterfactual questions.

### Cited Findings
- **Jev weaknesses (official list, no magnitudes):** literal reading; math and numbers; counting; score levels "weakly calibrated numerically"; dates; "double negatives and multi-hop reasoning reduce accuracy"; accuracy drops as irrelevant state grows ("context rot"); adversarial content in the state; contradictory instructions and criteria; **"Choice option order: leans toward the first option"** (mitigation: reorder options and check consistency); not a generator. — [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- TypeSafe design guidance: keep each question atomic, "a quick gut-check judgment". If a question needs extended reasoning or several factors, split it and combine the parts in code. — [TypeSafe docs](https://docs.typesafe.ai)
- **Kev** (dev / test; "new sources" means datasets not trained on): accuracy on new sources is Kev-0.8B 0.648/0.697, Kev-4B 0.817/0.838, Kev-9B 0.820/0.852, Kev-27B 0.851/0.889, Jev 0.857 (dev). Brier on new sources: 4B 0.269/0.242, 27B 0.225/0.156, Jev 0.211. MMLU-Pro: Kev-27B 0.675 vs Jev 0.840. Kev-9B ECE is 0.103 raw and 0.041 after temperature scaling. Share of decisions automatable at a 5% error budget: Kev 4B-27B 0.52-0.69, Jev 0.70. Known weaknesses: option order can change answers; knowledge questions depend on the base model; dates; context validated to 8,192 tokens (0.8B-9B). All numbers are the author's own ("verify on your own workload"). — [Kev README](https://github.com/jaredpalmer/kev)
- **Laya** on typed-decisions (400 cases, 2,000 decisions): fine-tuned 0.766 accuracy (Brier 0.062, ECE 0.213); **base English 0.362, below the majority baseline of 0.461** (random 0.318); Jev 1.13 (published) 0.727. Ordinal `score` questions are the weakest type (SST-5 0.372). Banking77 (77 labels): 0.425 vs Jev 0.870. Ships overconfident: refitting temperatures per type and option count cut mean ECE from 0.466 to 0.081. English context is 512 tokens. — [Laya card](https://huggingface.co/convaiinnovations/laya)
- **SemIf (Qwen3.5-4B, BF16):** 0.845 agreement on a 102-row TypeSafe subset vs Jev's published 0.883; 0.813 on 144 authored decisions (the EXL3 27B bridge scored 0.958); WANLI (NLI) 0.637. Temperature scaling took ECE from 0.208 to 0.069 on WANLI and changed little elsewhere. Most workloads are small and author-made, and the Jev score comes from TypeSafe records, not a live run. — [SemIf README](https://github.com/TheoLeeCJ/SemIf)
- **OpenJev:** no accuracy or calibration figures for DiffusionGemma itself, and no benefit measured for `think`. On JevBench (231 public items) OpenJev scored 87.4% vs JevK5's published 87.9%, with the same top answer on 230/231 items. Laya and Verdict cut long states silently; Laya's options share 256 tokens; CLM loses the start of long text and its score questions "may ignore the state". — [OpenJev README](https://github.com/razorback16/openjev)
- **NLI zero-shot:** mean f1_macro over the card's 28-task zero-shot table is 0.673-0.676 for the large v2.0 variants vs 0.497 for bart-large-mnli. ModernBERT-large is "slightly worse than DeBERTav3 on average" and "multiple times faster". — [deberta-v3-large-zeroshot-v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0); [ModernBERT-large-zeroshot-v2.0 card](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0)
- **gpt-6-luna Decisions:** no accuracy or calibration figures. The only performance claim is "about 10x faster than the Responses API". — [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
- **Safety classifiers:** ShieldGemma optimal F1 / AU-PRC is 0.825/0.887 (2B) and 0.830/0.883 (27B) on the SG Prompt set, vs GPT-4 at 0.810/0.847. Granite Guardian 3.3 harm F1 is 0.81 (no_think) and 0.79 (think). These are content-moderation benchmarks, not physical-injury ones. — [ShieldGemma card](https://huggingface.co/google/shieldgemma-2b); [Granite Guardian 3.3 card](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b)
- **Option-letter logprobs:** first-token vs text mismatch above 60% for some chat models (Llama2-7b-Chat). — [arXiv 2402.14499](https://arxiv.org/abs/2402.14499v1)

### Inferences
- Laya's 512-token English limit with silent truncation, and Verdict's silent cuts, matter for us. The full ASIMOV text prompt is about 6.2-7k characters (measured in `docs/apis/nli_zeroshot.md`, roughly 1.3k tokens), so the "full rubric" wording would be cut. A short state (scenario + instruction) fits. Kev's validated 8k, JevK5 and Clef's 16k, and Jev's 32k state limit fit the full prompt.
- Three published weaknesses line up with ASIMOV's questions. Q2 and Q4 are ordinal severity scales: Laya's score type is its weakest, and Jev's score levels are "weakly calibrated numerically". Asking them as `choice` rather than `score` avoids the score-type issue. Q3 and Q4 are counterfactual ("after the instruction"), and Jev's docs flag multi-hop reasoning and indirection. Q1 has 5 options with first-option bias. A permutation check (Kev has `/permute`) measures the order effect directly. These links are hypotheses for the owner to test, not documented results.
- Every vendor says to calibrate on your own labelled data. With 319 items and no training split, calibration could only be fitted by cross-validation on the eval set, which the hard rules forbid for training or tuning ("never train or tune on eval items"). That would count as tuning.

### Gaps
- Nothing published for any of these models on physical-safety, injury or counterfactual benchmarks. ASIMOV results would be new data.
- TypeSafe's own aggregate (the "711-row" figure SemIf mentions) and JevBench methodology were not found on the TypeSafe docs pages read.
- I found no magnitude for Jev's first-option bias.

## Q3. Cost, latency, parameters, hardware

### Takeaway
Hosted decision APIs bill input only and cost cents for the whole benchmark. Codiv's free tier covers it entirely. On a 16 GB M1 Pro, only small encoders (NLI, Laya), 4B logit scorers (SemIf / Qwen3-4B via MLX or MPS, about 8 GB bf16 or about 3 GB Q4) and Kev-0.8B run locally. Kev-4B/9B, OpenJev (about 16 GB at 4-bit "plus more in service") and the safety classifiers above 8B need hosted or cloud GPUs.

### Cited Findings
| System | Params | Price | Latency (author's figures) | Hardware | Source |
|---|---|---|---|---|---|
| Jev 1.13 | undisclosed | $0.042 / 1M input, output free | none published | hosted; 100K tok/s, 80 req/s; 64k per request, 32k state + longest question | [TypeSafe Models](https://docs.typesafe.ai/models) |
| gpt-6-luna Decisions | undisclosed | $0.10 / 1M input; no output or cache charges; regional and long-context multipliers | "about 10x faster than Responses" | hosted | [OpenAI guide](https://developers.openai.com/api/docs/guides/decisions) |
| Codiv (all 6 models) | see Q1 | Free: 100M System One tokens per account, no card; 60 req/min per key; paid tier not listed | OpenJev "tens of ms"; Laya ~10 ms; Verdict ~7 ms; CLM ~100 ms; JevK5 ~80 ms per question; Clef ~30 ms | hosted | [Codiv](https://codiv.ai) |
| OpenJev self-hosted | 26B-A4B | GPU time | RTX PRO 6000, 1 at a time p50: 30 ms (49-token state), 83 ms (2,047), 307 ms (8,191); MLX M3 Ultra / M4 Max 0.2-0.4 s per 3-question request | vLLM ≥24 GB GPU (~18 GB weights); MLX ~16 GB at 4-bit, grew to 23.5-36 GB in service | [OpenJev README](https://github.com/razorback16/openjev) |
| Kev-4B | ~4B | Modal L40S $1.95/h (H100 $3.95/h); fine-tune ~$1 | 41.5 ms on L40S, 18.1 ms on H100 (6 questions, short text); +~65 ms network; ~35 s cold start; Apple M5 721 ms for 5 questions | L40S/H100 or a **32 GB** Mac | [Kev README](https://github.com/jaredpalmer/kev) |
| Kev-0.8B | 0.8B | L4 $0.80/h | 22.7 ms | any 4 GB GPU or any Apple Silicon Mac | [Kev README](https://github.com/jaredpalmer/kev) |
| Laya 1.0 | 421M | free (Apache-2.0) or Codiv | T4: 39.5 ms for 1 question, 158.6 ms for 10; CPU 193-464 ms (multilingual) | T4 / CPU | [Laya card](https://huggingface.co/convaiinnovations/laya) |
| SemIf (Qwen3.5-4B) | 4B | free (MIT) | RTX 3090: 21 binary criteria in 1.023 s (vs 5.332 s generating); 2.33 decisions/s fresh, 20.03/s parallel suffixes | 4B BF16 GPU; MLX/MPS; llama.cpp Q4_K_M ~3.0 GB | [SemIf README](https://github.com/TheoLeeCJ/SemIf) |
| NLI zeroshot-v2.0 | ~435M (DeBERTa-v3-large) | free | ModernBERT ~1,100 texts/s on A100 (short texts) | CPU-feasible, ~0.8-0.9 GB | [ModernBERT card](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) |
| gpt-oss-safeguard-20b | 21B (3.6B active) | free weights | none | 16 GB VRAM | [card](https://huggingface.co/openai/gpt-oss-safeguard-20b) |
| ShieldGemma | 2B / 9B / 27B | free weights | none | — | [card](https://huggingface.co/google/shieldgemma-2b) |
| Granite Guardian 3.3 | 8B | free weights | none ("moderate cost, latency") | — | [card](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b) |
| Llama Guard 4 | 12B | free weights / hosted providers | — | — | [Meta docs](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama-guard-4/) |

### Inferences
- Rough cost for 319 items × ~1.5k input tokens ≈ 0.5M tokens (my arithmetic, **UNVERIFIED** token counts): Jev ≈ $0.02, Decisions ≈ $0.05, Codiv $0 (0.5% of the free 100M). OpenJev with `think` bills input twice plus the thought as output; Codiv's price for output tokens isn't listed.
- On a 16 GB M1 Pro: gpt-oss-safeguard-20b is documented for 16 GB of VRAM, but unified memory shared with the OS makes it tight (**UNVERIFIED**). ShieldGemma-2B, Laya, NLI and SemIf Q4 are the comfortable local options. Kev-4B is documented as needing a 32 GB Mac.
- Latency comparisons across these sources aren't apples-to-apples: they use different GPUs, state lengths, question counts, and model time vs end-to-end time. For H1, latency needs measuring under one harness.

### Gaps
- No public latency figures for Jev or gpt-6-luna Decisions. Codiv's paid pricing isn't listed.
- No M1 Pro measurements for any of these.

## Q4. When a decision model beats an LLM and when it doesn't

### Takeaway
Vendors position decision models for fast, cheap, calibrated "gut-check" judgments over a given state with a small set of options (routing, triage, moderation). Their own docs advise against them for multi-hop reasoning, arithmetic and counting, long or irrelevant context, and many-option or ordinal-magnitude questions. They suggest either splitting the question into atomic parts or escalating low-confidence cases to a reasoning model. No source addresses counterfactual "what if the robot does X" questions directly.

### Cited Findings
- System One models "do not write replies, produce code, or generate explanations of their reasoning". Confidence lets you decide "when to act and when to escalate to a person or a reasoning model". Calibration "does not guarantee that an individual answer is correct". — [TypeSafe System One](https://docs.typesafe.ai/concepts/system-one)
- Split questions that need extended reasoning or several factors into atomic questions and combine them in code. — [TypeSafe docs](https://docs.typesafe.ai)
- Jev: multi-hop reasoning and double negatives reduce accuracy; irrelevant state causes "context rot"; there is first-option bias. — [Jev jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- OpenAI: use Decisions for "a probability, a category choice, or a rubric score". Use Structured Outputs or Responses for written explanations or custom JSON. Decisions that depend on earlier answers need separate requests. — [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
- Kev is a good fit for self-hosting, triage and routing, fine-tuning on labelled examples, and low per-request cost. It does worse on knowledge-heavy questions, date math, and automation that needs sharp confidence ranking. — [Kev README](https://github.com/jaredpalmer/kev)
- SemIf works for runtime-defined routing, triage and evidence checks with few options. It is weaker on hard NLI (WANLI 0.637). On 21 criteria, direct logits and generation chose the same option on 18 of 21, with generation about 5.2× slower. — [SemIf README](https://github.com/TheoLeeCJ/SemIf)
- OpenJev adds a `think` budget specifically for "multi-step problems" (512+ tokens), with no measured gain. — [OpenJev README](https://github.com/razorback16/openjev)
- Laya: base checkpoints need fine-tuning; accuracy degrades with high option counts. — [Laya card](https://huggingface.co/convaiinnovations/laya)
- For instruction-tuned LLMs, reading option probabilities from the first token can misstate what the model would answer in text, and text answers are more robust to perturbations when the two disagree. — [arXiv 2402.14499](https://arxiv.org/abs/2402.14499v1); [arXiv 2404.08382](https://arxiv.org/abs/2404.08382)
- ShieldGemma is "highly sensitive to how the safety principle is worded". Granite custom criteria "require testing". — [ShieldGemma card](https://huggingface.co/google/shieldgemma-2b); [Granite Guardian card](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b)

### Inferences (facts for the owners to weigh, not conclusions)
- For ASIMOV text, the vendor guidance points to different expectations per question type. Q1 (injury type, 5 nominal options) is the most System-One-shaped. Q3 and Q4 depend on reasoning about the instruction's effect, which is what the docs list as a weakness and what OpenJev's `think` targets. Comparing `think` 0 vs 512 on Q3/Q4 is a direct test of that guidance. Whether the gap appears is an empirical question.
- The ASIMOV rubric is long. TypeSafe's "context rot" and atomic-question advice suggest that putting the full rubric in `state` may cost accuracy compared with putting only the relevant rubric lines in each question's `instructions`/`criteria`. This is the wording-rule choice the owner already has open (PLAN §4.3).
- Option-order sensitivity is documented for Jev and Kev and for LLM logprob scoring generally. One permuted rerun per system would show how much this affects the comparison.

### Gaps
- No primary source compares decision models to generative LLMs on counterfactual or hypothetical safety questions, or on long-rubric ordinal grading.
- No published accuracy gain from OpenJev's `think` or from gpt-6-luna Decisions vs Responses on any benchmark.
