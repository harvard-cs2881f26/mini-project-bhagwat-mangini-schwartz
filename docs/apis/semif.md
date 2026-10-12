# semif: SemIf (formerly OpenJev), open "semantic if" decision scorer

## 1. What it is

SemIf is an open-source (MIT) project by TheoLeeCJ that copies the *interface pattern* of
TypeSafe's closed Jev decision service using open models. It does not copy Jev's model or
training. The input is a state, a question and 2-16 described options. One forward pass of a
frozen LLM (reference: `Qwen/Qwen3.5-4B`) reads the logits of fixed uppercase answer tokens, and
a softmax over those logits gives a probability per option. Nothing is decoded and no output
tokens are produced. The README says it is "not affiliated with or endorsed by TypeSafe".
Sources, read 2026-10-12:
- Repo + README: https://github.com/TheoLeeCJ/SemIf (branch `master`, last push 2026-09-23, MIT)
- `docs/METHOD.md` and `results/raw/shape777-direct.predictions.jsonl` in the same repo
- CPU port (third-party): https://huggingface.co/Meanblock/JEV-CPU

**Name overlap, check with Noa:** the README title is "SemIf (formerly OpenJev)", and it points to
the site openjev.com. PLAN §4 lists both `semif` and `openjev_think` as separate optional systems.
PLAN also lists OpenJev as a `/v1/systemone` client config row (§5 Step B, §9 table). That `/v1/systemone`
OpenJev may be a different, hosted project. Confirm what `openjev_think` refers to before building both.

## 2. Access

- **Local (documented):** public code, MIT. No weights are bundled; Qwen3.5-4B is Apache-2.0 per its
  card (third-party summary, not read directly). No auth, no cost.
- **Hosted (UNVERIFIED):** a third-party blog
  (https://explainx.ai/blog/semif-langsmith-gateway-free-week-decision-ai-2026) says the LangSmith
  LLM Gateway hosts it as `semif-qwen3.5-4b`, free until **2026-09-28**, which has passed. The
  official page https://docs.langchain.com/langsmith/llm-gateway-decision-models.md lists only
  `typesafe/jev-1.13.0` and OpenAI decision models, not SemIf. Do not plan on the hosted route
  unless it is confirmed.

## 3. Endpoint / interface

There is no server or Python API in the README. Interface is a CLI over JSONL (copied from README):
```bash
CUDA_VISIBLE_DEVICES=0 semif-score \
  --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl \
  --output results.jsonl
```
- Modes: `direct` (fresh per row), `shared` (the README says use it "If every row has the same exact
  state": the state is processed once and the criteria run in parallel). The reranker mode is a comparison baseline.
- Backends: CUDA (default), `--backend mlx` (Apple Silicon), `--device mps`,
  `--backend llamacpp --gguf <file>` (CPU), and `--device cpu --dtype float32` ("much slower").
- **Input row:** `id`, `state` (string or a non-empty JSON object/array), `question`, `options`
  (list of `{id, description}`). METHOD.md allows 2-16 options.
- **Output row** (fields seen in a committed results file, not a documented schema): `id`, `option_ids`,
  `probabilities` (aligned with `option_ids`), `option_logits`, `allowed_token_mass`,
  `input_tokens`, `prompt_sha256`, `prompt_version` (e.g. `direct-options-v1`), `model{source,
  revision, dtype, ...}`, `readout`. The README adds: "Each result contains typed option scores,
  timing, the exact model revision, and a prompt hash."

## 4. For our use

- **One option per answer, with a probability per option: yes.** This is the design goal. The probabilities are
  "conditional on the supplied options" and uncalibrated (temperature-scaling tools are in the repo).
- **One row per question.** Each item gives 4 rows sharing one `state`: the scenario +
  instruction, or the full verbatim prompt (PLAN §4.3 wording rule; the owner decides). Option `id`s =
  letters, `description` = option text. `--mode shared` fits a run where all rows share one state.
  Whether shared mode also works on a mixed file grouped by state is **UNVERIFIED**.
- `allowed_token_mass` (probability mass on the answer tokens) is a built-in check on
  whether the model is answering as instructed. Worth logging.
- This overlaps with `qwen_logprob` (PLAN §4): same idea (option logits from a small frozen Qwen),
  different prompt and model version (Qwen3.5-4B vs Qwen3-4B). The contrast it adds is small.
  Noa should decide whether it earns a slot.

## 5. Limits and errors

- Max input: not stated (**UNVERIFIED**). It is limited by the base model's context. Committed rows
  show `input_tokens` around 1,890, and METHOD.md uses states of about 8,000 characters, so our
  ~1.3k-token prompts should fit (**UNVERIFIED** for our data).
- Error behaviour and codes are not documented. The README warns that the BF16 fast reuse paths changed 5-6 of
  777 argmaxes compared with fresh scoring, and that llama.cpp gives small numerical differences.
  Compare with a tolerance.

## 6. Example

Input (copied from the README):
```json
{"id": "route-1", "state": "Customer cannot access an account after a password reset.",
 "question": "Which queue should handle this request?",
 "options": [{"id": "access", "description": "Account access support."},
             {"id": "billing", "description": "Billing support."}]}
```
Output (one row copied from `results/raw/shape777-direct.predictions.jsonl`, trimmed):
```json
{"mode": "fresh", "id": "015c14ff8f2bce2886b3", "option_ids": ["yes", "no"],
 "probabilities": [0.4378, 0.5622], "option_logits": [25.0, 25.25],
 "allowed_token_mass": 0.99968, "input_tokens": 1891, "prompt_version": "direct-options-v1",
 "model": {"source": "Qwen/Qwen3.5-4B", "revision": "851bf6e8...", "dtype": "bfloat16"}}
```

## 7. Hardware

- Install: clone the repo, then `pip install -e '.[test]'` (CUDA), `'.[test,mlx]'` (Apple Silicon) or
  `'.[test,llamacpp]'` (CPU + GGUF, e.g. `bartowski/Qwen_Qwen3.5-4B-GGUF` Q4_K_M, about 3.0 GB).
- GPU: "a GPU that can hold a 4B BF16 model" (reference: RTX 3090). About 8-10 GB of VRAM is
  **UNVERIFIED**; the README gives no number.
- This Mac: the MLX or MPS path is documented. Speed is **UNVERIFIED**. RTX 3090 reference: 21 binary
  decisions in 1.023 s (direct), and 2.33 decisions/s fresh on about 8k-character states.
- Reported quality: 0.845 agreement on a 102-row TypeSafe subset vs Jev's published 0.883 (README).

## 8. Status

`READY` (local CLI). Identified with confidence and documented well enough to wrap. Two checks with Noa
before building: (1) is `openjev_think` the same project (SemIf was formerly OpenJev)? (2) Is it
different enough from `qwen_logprob` to keep? The hosted LangSmith route is UNVERIFIED and
may have expired.
