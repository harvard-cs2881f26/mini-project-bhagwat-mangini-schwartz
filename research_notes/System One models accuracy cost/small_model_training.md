# Improving small models (0.4B–4B) on the ASIMOV-2.0 text benchmark without training on eval items

Researched 2026-10-12. Scope: the **text** benchmark (paper: ASIMOV-2.0-Injury, 319 items, 4 multiple-choice questions). These notes are options and sources only. Which experiment to run, and what the results mean, is for the H1 owner to decide (CLAUDE.md hard rule).

## 1. Training data that is not the eval set

### Takeaway
No public **labeled** training split exists for ASIMOV-2.0 text. Every public ASIMOV-2.0 build is a single `val` split, and the 2.3M-instruction ASIMOV v1 "Injury train" set from the v1 paper is **not** in the public bucket. The realistic sources are: (a) raw NEISS injury narratives from years other than 2023, used as seeds for teacher-labeled synthetic data, (b) ASIMOV v1 `asimov_injury_val` (304 items, NEISS 2023, with injury fields) after dedup against the eval set, and (c) other physical-safety sets for extra signal (SafetyBench Physical Health, SafeAgentBench). None of these has labels for all four of our questions.

### Cited Findings
- ASIMOV-2.0 text items were built from NEISS: "We took narratives from 2023 data with rebalancing across NEISS diagnostic codes." 319 scenarios were kept, each with answers from 5 human raters, and low-consensus items were filtered out. The paper mentions no released train split. — [ASIMOV-2.0 paper, arXiv 2509.21651 (HTML v2)](https://arxiv.org/html/2509.21651v2)
- The ASIMOV-2.0 paper's only post-training targets the **image/Constraints** task: 200 constraint-image pairs, SFT then PPO, on GR-ER-1.5. It is not about the text/Injury task. The paper says nothing about releasing that training data. — [arXiv 2509.21651v2](https://arxiv.org/html/2509.21651v2)
- Label space (paper Tables 2–4). Latent severity: Severe 139 / Moderate 142 / Minor 32 / None 6. Activated severity: 108 / 90 / 78 / 43. Effect: same 121, reduces 97, eliminates 60, increases 41. — [arXiv 2509.21651v2](https://arxiv.org/html/2509.21651v2); the repo's `data/README.md` has the matching counts. Q1 (injury type) options differ per item: four injury types plus "No realistic injury possible" (repo `data/README.md`).
- **Public bucket listing (checked 2026-10-12, `storage.googleapis.com/storage/v1/b/gresearch/o?prefix=robotics/asimov`).** Builds present: `asimov_v2_injuries` (319, val only, features `prompt`, `answer`), `asimov_v2_videos`, `asimov_v2_constraints_{with,without}_rationale`, `asimov_injury_val`, `asimov_dilemmas_auto_val`, `asimov_dilemmas_scifi_val`, `asimov_dilemmas_scifi_train` (~446 MB, but its only split file is named `-val`), and `asimov_multimodal_{auto,manual,robopair}_val`. **There is no `asimov_injury_train` object.** — [GCS JSON listing](https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=robotics/asimov)
- The ASIMOV v1 paper (Table 1) reports an "ASIMOV-Injury" **train** subset of about 361,947 contexts and 2,335,361 instructions. None of the v1 train subsets has human labels. Total across subsets: 513,679 contexts and 2,942,060 instructions. — [Sermanet et al., "Generating Robot Constitutions & Benchmarks for Semantic Safety", arXiv 2503.08663](https://arxiv.org/pdf/2503.08663) (via search summary; full PDF too large to fetch, so **UNVERIFIED** in detail). This train set does not appear in the public bucket above.
- The TFDS catalog lists `asimov_dilemmas_scifi_train` as split 'val' with 9,004 examples, while the paper reports 9,056. The sources disagree. — [TFDS catalog](https://www.tensorflow.org/datasets/catalog/asimov_dilemmas_scifi_train)
- **`asimov_injury_val` (ASIMOV v1)** has one 'val' split of 304 examples. Its features include `context`, `instruction`, `undesirable_groundtruth_answer` (binary), several prompt variants (with/without constitution, chain of thought), and the raw NEISS record: `Narrative_1`, `Diagnosis`, `Diagnosis_2`, `Disposition`, `Body_Part`, `Product_1..3`, `Age`, `Location`, `CPSC_Case_Number`, `Treatment_Date`. — [features.json](https://storage.googleapis.com/gresearch/robotics/asimov_injury_val/0.1.0/features.json), [dataset_info.json](https://storage.googleapis.com/gresearch/robotics/asimov_injury_val/0.1.0/dataset_info.json), [TFDS catalog](https://www.tensorflow.org/datasets/catalog/asimov_injury_val)
- ASIMOV v1 also drew its injury instances from NEISS 2023 (338,265 cases) and generated them with Gemini 1.5 Pro. — [arXiv 2503.08663](https://arxiv.org/pdf/2503.08663) (via search summary, **UNVERIFIED** detail)
- **NEISS raw data** comes as annual files (Excel, tab-delimited, SAS), each covering one calendar year; the most recent 20 years are online, and a new year usually appears the following April. Narrative length rose from 142 to 400 characters for treatment dates from 2019 on. Fields include narrative, diagnosis (plus Diagnosis_2 from 2018), body part, product codes, and alcohol/drug involvement. — [CPSC NEISS Injury Data](https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data). Disposition codes, which are a plausible severity proxy, are in the NEISS data dictionary; I did not open it (**UNVERIFIED** code list). License terms are not stated on that page.
- **SafetyBench** (thu-coai, MIT license) has about 11,400 English and Chinese multiple-choice questions in 7 categories, one of which is **Physical Health**. Test answers were open-sourced in July 2025. — [HF thu-coai/SafetyBench](https://huggingface.co/datasets/thu-coai/SafetyBench/tree/1b767ac2cd3e25c4b4f317278941e7f8a58028e2/dev); [arXiv 2309.07045](https://arxiv.org/abs/2309.07045). How to filter the Physical Health items was not confirmed.
- **SafeAgentBench** has 750 household tasks (450 hazardous, 300 safe) covering 10 hazard types. — [arXiv 2412.13178](https://arxiv.org/html/2412.13178v5). **IS-Bench** has 161 scenarios with 388 safety risks in OmniGibson. — [arXiv 2506.16402](https://arxiv.org/html/2506.16402v3). **RiskAwareBench** — [arXiv 2408.04449](https://arxiv.org/html/2408.04449v1). All three are framed as planning tasks, not injury-type/severity labels.

### Inferences
- The only way to get training data in the exact 4-question format is **synthetic**: take NEISS narratives, have a large teacher write a home scenario plus a robot instruction in the ASIMOV-2.0 style, and have the teacher (ideally more than one) answer the 4 questions. NEISS Diagnosis could seed the injury-type options and Disposition could cross-check severity. This mirrors how the paper built the eval set.
- **Leakage precaution, structural:** the eval set uses NEISS **2023**. Seeding from other years (e.g. 2019–2022 and 2024) rules out reusing the same case. ASIMOV v1 `asimov_injury_val` is also NEISS 2023, so its narratives could be the same cases behind eval items. The v2 build has no `CPSC_Case_Number`, so overlap can only be checked by text.
- **Leakage precaution, textual:** before training, compare every training scenario to all 319 eval prompts using character n-gram/MinHash Jaccard and embedding cosine. Drop anything above a threshold, and report how many were dropped. Pick the threshold **before** looking at which eval items match, and log it in WORKLOG. Also keep eval items out of the teacher's few-shot examples.
- The eval label distribution (e.g. latent severity mostly Moderate/Severe) is itself information about the eval set. Using it to set priors, class weights or synthetic-data balance would be tuning on eval. Balance synthetic data from NEISS's own distribution instead.
- The SafetyBench and SafeAgentBench formats don't match ours. They are more useful as auxiliary tasks or a sanity check than as the main training set.

### Gaps
- Whether Google DeepMind ever published the v1 `asimov_injury_train` elsewhere (it is not in `gresearch/robotics`). Not found.
- How many candidate scenarios the paper generated before filtering to 319, and its rater/quality filter thresholds. The paper doesn't say.
- The NEISS Disposition code meanings and the license were not fetched.

## 2. Distillation recipes and how Kev and Laya were trained

### Takeaway
Two working templates exist. **Kev:** a frozen Qwen3.5 base + rank-16 LoRA + pointer head, trained with cross-entropy on the right option, about 1 h on one H100 for 4B. **Laya:** a fully fine-tuned ModernBERT-large (421M) + decision head, trained with RL on proper scoring rules. Laya's own card shows that the base model is below the majority baseline zero-shot and only becomes competitive after **in-domain** fine-tuning. Rationale distillation ("Distilling step-by-step") helps small models, but models of 3B or less learn better from **short** rationales than from long chains of thought.

### Cited Findings
- **Kev:** Kev-0.8B/4B/9B = Qwen3.5 base (frozen) + LoRA rank 16 on attention, MLP and DeltaNet projections + a small pointer head, cross-entropy loss on the correct answer. Base data `decision-v7`: 10,000 examples from ten public datasets + 896 generated policy examples + 1,680 examples from 60 generated rule structures. 2 epochs, LR 1e-4 (0.8B) / 5e-5 (4B, 9B). Follow-up fine-tunes use `--init_from`. — [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
- Kev compute: the first stage took about 20 min on one H100 for 0.8B and about 1 h on one H100 (via Modal) for 4B. "No Jev outputs were used for training." The README doesn't say which LLM generated the synthetic data. — [Kev README](https://github.com/jaredpalmer/kev)
- Kev results (accuracy, dev/test, new sources): 27B 0.851/0.889, 9B 0.820/0.852, 4B 0.817/0.838, 0.8B 0.648/0.697, Jev 0.857/–. — [Kev README](https://github.com/jaredpalmer/kev)
- Kev fine-tuning gains reported by users: a support workload went from 67.7% to 73.6% (4B), and 5,219 consumer-finance complaints went from 0.804 to 0.904 in-distribution. Fine-tuning from the raw base scored 0.33 on Kev's eval, versus 0.84 for the released model. `--init_from` kept 0.83 and reached 0.88 on the new domain. — [Kev README](https://github.com/jaredpalmer/kev). Lesson: start from the released Kev adapter, not the base model, to avoid forgetting.
- Kev hygiene: checkpoints were chosen on dev sets, and each test set was read once. Calibration temperatures were fit on dev (0.8B/4B) or on held-out datasets (9B/27B). — [Kev README](https://github.com/jaredpalmer/kev)
- **Laya:** ModernBERT-large (fully fine-tuned) + a 2-layer decision head with an option-marker scorer and an act/escalate head, about 421M parameters (the multilingual version is mmBERT-base, 322M). Trained with "RLCD" (Reinforcement Learning for Calibrated Decisions): rewards from strictly proper scoring rules, REINFORCE with a group-mean baseline. The card doesn't say what data the base checkpoints were trained on or where the labels came from. — [HF convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
- Laya on its typed-decisions benchmark (2,000 decisions): fine-tuned `laya-typed-decisions` 0.766 ("fine-tuned on that benchmark's own training split"), base English 0.362, base multilingual 0.342, majority class 0.461, random 0.318, Jev 0.727 (published, not measured by them), teacher self-agreement ceiling 0.735. The fine-tuning notebook runs on Kaggle's 2× T4. — [Laya card](https://huggingface.co/convaiinnovations/laya)
- **Distilling step-by-step** (Hsieh et al. 2023): LLM rationales are used as extra supervision in a multi-task setup. A 770M T5 beat few-shot 540B PaLM on one benchmark using 80% of the data, while standard fine-tuning didn't match it even with 100%. — [arXiv 2305.02301](https://arxiv.org/abs/2305.02301)
- **Small Model Learnability Gap** (Li et al. 2025): models of 3B or less don't reliably gain from long chain-of-thought data or from distilling larger models. "Mix Distillation" (mixing long and short CoT, or large- and small-teacher outputs) gave Qwen2.5-3B-Instruct more than 8 points on MATH/AMC (Mix-Long) and more than 7 points (Mix-Large). — [arXiv 2502.12143](https://arxiv.org/abs/2502.12143)

### Inferences
- A distillation pipeline for us: (1) generate synthetic NEISS-seeded items (section 1); (2) label them with 1–3 strong teachers, keeping per-option **probabilities** where the API gives them (soft labels) and a short rationale; (3) keep only items where the teachers agree, or train on averaged soft labels; (4) train either a Kev-style LoRA + option-scoring head on Qwen 0.8B–4B, or a ModernBERT-large/DeBERTa-v3 multiple-choice head (one forward pass per option, about 0.4B parameters, fast on a Mac).
- Teacher-agreement on the synthetic data is a ceiling: Laya's card reports "teacher self-agreement" of 0.735. The eval labels are a 5-rater human consensus, so teacher bias carries over to the student.
- The four questions can share one encoder with four heads (multi-task), or be asked one at a time. Kev's `/v1/systemone` already treats them as separate questions (docs/apis/systemone.md).
- No source gives expected gains on ASIMOV-style injury severity specifically. Gains from Kev and Laya are on their own decision benchmarks and in-domain, so they are an upper-bound analogy, not a prediction.

### Gaps
- Which teacher and prompts generated Kev's and Laya's synthetic data, and Laya's base training corpus: not disclosed.
- No published work fine-tunes small models on ASIMOV (v1 or v2) injury/severity questions. Not found.

## 3. LoRA / QLoRA on a 16 GB M1 Pro with MLX, and on cheap cloud GPUs

### Takeaway
`mlx-lm` supports LoRA, DoRA and full fine-tuning, and automatically switches to QLoRA when given a quantized model. On 16 GB, the settings are batch size 1–2, 4–8 layers, `--grad-checkpoint`, short sequences, and 4-bit weights for anything above about 1.5B. There is no published tokens-per-second figure for a 16 GB M1 Pro: the only official number is about 250 tok/s for Mistral-7B on a 32 GB M1 Max. Kev-4B itself needs a 24 GB GPU or a 32 GB Mac to serve, so 4B training on this Mac is borderline. Modal is cheap enough for a few 1-hour runs inside the $30/month free credit.

### Cited Findings
- `mlx_lm.lora --model <hf or local> --train --data <dir> --iters N`. Data is `train.jsonl` / `valid.jsonl` / `test.jsonl` in `chat`, `completions` (`prompt` / `completion`), `text` or `tools` format, one example per line. `--mask-prompt` computes loss only on the completion. A quantized model triggers QLoRA. `mlx_lm.fuse` merges the adapter, and `--test` reports test perplexity. — [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- Memory tips from the docs: quantize (`convert -q`), lower `--batch-size` (default 4) and use `--grad-accumulation-steps`, lower `--num-layers` (default 16; try 8 or 4), split long examples, `--grad-checkpoint`. The only speed figure: Mistral-7B, batch size 1, 4 layers, on a 32 GB M1 Max, "about 250 tokens-per-second". — [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- Community 16 GB example: `--batch-size 2 --num-layers 8 --rank 4` on a 4-bit model. — [Towards Data Science, "Local LLM fine-tuning on Mac M1 16GB"](https://towardsdatascience.com/local-llm-fine-tuning-on-mac-m1-16gb-f59f4f598be7) (secondary source; settings **UNVERIFIED** by me)
- Kev-4B serving requirement: a 24 GB GPU or a 32 GB Mac (repo `docs/apis/systemone.md`, citing the Kev README).
- Modal GPU prices (per second, converted to per hour): T4 $0.59, L4 $0.80, A10 $1.10, L40S $1.95, A100-40GB $2.10, A100-80GB $2.50, H100 $3.95, H200 $4.54, B200 $6.25. The Starter plan includes $30/month of free compute. — [modal.com/pricing](https://modal.com/pricing)

### Inferences (rough estimates, not measured)
- At an assumed 300-token prompt+answer per item and 5,000 synthetic items, one epoch is about 1.5M tokens. At 250–1,000 tok/s on the Mac (0.5B–1.5B models should beat the 7B figure, but an M1 Pro is slower than an M1 Max), that is about 25–100 min per epoch. A 4B 4-bit QLoRA would probably be several times slower and close to the memory limit. **Measure it with a 50-iteration run first:** `mlx_lm.lora` prints tok/s.
- Kev-4B's first stage (about 1 h on an H100, roughly 12k examples × 2 epochs) costs about $4 on Modal at $3.95/h. A comparable 4B LoRA on our data would be a few dollars, well inside the free $30. Per CLAUDE.md, ask before any paid run.
- A ModernBERT-large classifier (Laya-size) trains comfortably on a T4/L4 (Laya's notebook uses 2× T4) or on MPS, and is the cheapest option at inference.
- mlx-lm's standard LoRA trains generation (next-token loss). An option-scoring head like Kev's needs custom code. A simpler route that fits mlx-lm: train the completion to be the answer letter or the JSON with `--mask-prompt`, then read option-letter log-probs at inference.

### Gaps
- No measured mlx-lm LoRA throughput or peak memory for Qwen 0.5B/1.5B/3B/4B on a 16 GB M1 Pro was found.
- Whether mlx-lm supports Qwen3.5 (DeltaNet layers, the Kev base) was not checked.

## 4. Prompting-only improvements for small models

### Takeaway
Few sources speak directly to small models on this kind of multiple-choice safety classification. The evidence suggests: (a) long chain-of-thought helps small (3B or less) models less than large ones; (b) safety-aware CoT can help on embodied safety but has trade-offs; (c) few-shot examples must come from non-eval data, i.e. the synthetic or v1 items from section 1. The paper's own small-model gap is large, which leaves room for improvement.

### Cited Findings
- Fig. 4 small-model gap: "GPT5-Mini and GPT5-Nano are 20% and 5% lower in latent risk accuracy" than GPT-5, and 19% / 17% lower on high-severity accuracy. On action safety, smaller models were "substantially weaker, particularly for GPT-5-Mini, GPT-5-Nano and Gemini 2.5 Flash-lite", with Gemini 2.5 Flash the exception. Top models reach 74% and 66% on these metrics. Gemma is not evaluated in the text. — [arXiv 2509.21651v2](https://arxiv.org/html/2509.21651v2)
- IS-Bench: safety-aware chain-of-thought improves safety but "often compromises task completion". — [arXiv 2506.16402](https://arxiv.org/html/2506.16402v3)
- Models of 3B or less learn better from short reasoning chains. — [arXiv 2502.12143](https://arxiv.org/abs/2502.12143). This is about training, not prompting. Its application to CoT prompting is an inference.
- ASIMOV v1 shipped prompt variants with a constitution and with constitution + chain of thought, so "constitution" prompting is a documented ASIMOV-family technique. — [asimov_injury_val features.json](https://storage.googleapis.com/gresearch/robotics/asimov_injury_val/0.1.0/features.json); the v1 paper reports up to 84.3% alignment with generated constitutions — [arXiv 2503.08663](https://arxiv.org/pdf/2503.08663) (search summary, **UNVERIFIED** detail)

### Inferences (candidate variants; each would go in `experiments/` under its own run name, per CLAUDE.md)
- **Few-shot:** 2–8 worked examples drawn from synthetic NEISS-non-2023 items or dedup-checked v1 items, never from the 319.
- **Rubric simplification:** spell out severity definitions (e.g. map to NEISS disposition: treated and released vs admitted vs fatal) and the four "effect" categories. A rubric written from NEISS, not from eval answers, is not tuning on eval.
- **Ask one question at a time** instead of all four in one JSON. This cuts format failures for small models and allows option-log-prob scoring with no parsing.
- **Self-consistency:** sample k answers and take the majority vote. Cost grows linearly with k, which matters for H1's accuracy-per-dollar framing. Plot accuracy against k.
- **CoT vs direct:** for 0.5B–4B models, short CoT or none is the expected sweet spot (inference from 2502.12143). Test it rather than assume it.
- Any prompt variant chosen by looking at eval accuracy is selection on eval. Choose variants on a synthetic dev split, then run the chosen ones on the 319 once.

### Gaps
- No source measured self-consistency or few-shot gains for 0.5B–4B models on physical-safety multiple choice.
- No exact Fig. 4 values for Flash or Flash-Lite in the paper text (they are in figure images).

## 5. Evaluation hygiene

### Takeaway
Use the Kev pattern: choose everything on a dev set, read the test (our 319) once per final model, and report contamination checks and the teacher ceiling next to accuracy and cost.

### Cited Findings
- Kev: checkpoints selected on dev sets, test read once, calibration fit on dev or held-out data. Its README says outright that the Jev comparison isn't controlled because Jev's training data is unknown, and that Kev-27B's contamination status "can't be assessed". — [Kev README](https://github.com/jaredpalmer/kev)
- Laya's headline 0.766 is in-domain (trained on that benchmark's train split), and the card has no contamination analysis. — [Laya card](https://huggingface.co/convaiinnovations/laya)
- Eval labels are a 5-rater consensus with low-consensus items filtered out. — [arXiv 2509.21651v2](https://arxiv.org/html/2509.21651v2)

### Inferences
- Splits: synthetic data → train / dev (choose hyperparameters, prompts, calibration) → the 319 eval items as test, scored through `scripts/run.py` only. Optionally keep a second, human-checked synthetic test set to show that gains generalize beyond the teacher.
- Report: dedup method + threshold + number dropped; NEISS years used; the teacher(s) and their accuracy on the 319 (the "teacher ceiling"); baselines (majority, random, untrained base model, same model with the paper prompt); and per-question accuracy, parse/refusal rates, latency, tokens, parameters and $ per model.
- Pretrained-model contamination is a separate risk: ASIMOV-2.0 has been public since September 2025, so newer base models or teachers may have seen it. A cheap check: prompt the model with the start of an eval scenario and see whether it reproduces the rest.
- An off-the-shelf model already tuned by someone else (Kev-27B, Laya base) has unknown data. Its results are comparable only as "off-the-shelf", not as "not trained on eval".

### Gaps
- No ASIMOV-specific contamination study exists. Not found.
