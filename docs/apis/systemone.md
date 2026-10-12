# `/v1/systemone` (Jev, Kev, Open-Jev, Laya)

Read 2026-10-12. Facts below come from the official TypeSafe docs or each project's own
README / model card unless marked **UNVERIFIED**. No endpoint was called.

## 1. What it is

`POST /v1/systemone` is TypeSafe's "System One" API: you send a `state` (text or JSON) and a
map of typed questions, and you get back typed answers with probabilities. No text is generated.
**Jev** is TypeSafe's closed hosted model (current `jev-1.13.0`). Three open projects serve the
same request/response shape: **Kev** (jaredpalmer, LoRA + pointer head on Qwen3.5),
**Open-Jev** (Zefan-Cai, LoRA + scalar head on Qwen), **Laya** (convaiinnovations, ModernBERT
encoder). Sources: https://docs.typesafe.ai/api, https://docs.typesafe.ai/models,
https://github.com/jaredpalmer/kev, https://github.com/Zefan-Cai/Open-Jev,
https://huggingface.co/convaiinnovations/laya.
"OpenJev" is ambiguous: TheoLeeCJ/openjev (now "SemIf", MIT) and SiliconLabAI/OpenJev are
different projects. I took **Zefan-Cai/Open-Jev** because it serves `/v1/systemone` and lists
`jev-latest` as an alias. **UNVERIFIED** that this is the one PLAN.md means.

## 2. Access

| System | Access | Auth | Price |
|---|---|---|---|
| Jev | Paid API, key from https://console.typesafe.ai/keys | `Authorization: Bearer <key>`; SDK env var `TYPESAFE_API_KEY` (base URL override `TYPESAFE_BASE_URL`) | $0.042 per M input tokens, output free |
| Kev | Open weights, Apache-2.0 (HF `jaredpalmer/kev-0.8b/4b/9b/27b`, tag `v1.0`) | none by default; set `KEV_API_KEY` to require Bearer | free (own hardware) |
| Open-Jev | Open weights (LoRA packages, MIT code), HF `ZefanCai/Open-Jev-2B/9B/27B-v1.1`; Qwen base weights fetched separately | none (binds to loopback) | free |
| Laya | Open weights, Apache-2.0, `pip install "laya[serve]"` | **UNVERIFIED** | free |

Secondary sites say Jev is also served via OpenRouter, Vercel AI Gateway and LLM Gateway
(`https://api.llmgateway.io/v1/systemone`): **UNVERIFIED**, not needed.
No free tier or trial credit found in the TypeSafe docs.

## 3. Endpoint / interface

- Jev: `POST https://api.typesafe.ai/v1/systemone`, `Content-Type: application/json`.
  `GET /v1/models` lists the names your account can send.
- Kev: `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009` ->
  `http://127.0.0.1:8009/v1/systemone`, model `kev-latest`. Extra endpoints:
  `/v1/systemone/permute` (one Choice over 1-64 option orders) and `/v1/systemone/separate`.
- Open-Jev: `python -m jev.server --checkpoint ...` -> `http://127.0.0.1:8791/v1/systemone`.
  `/v1/models` lists aliases `open-jev`, `jev-latest`. Its example request has no `model`
  field; whether it rejects `jev-1.13.0` is **UNVERIFIED** (I did not read `predictor.predict`).
- Laya: `laya-serve` on `localhost:8000/v1/systemone`.

**Request** (TypeSafe API reference):
- `state` (required): string, object or array.
- `model` (required for Jev): `jev-latest` (-> `jev-1.13.0`), `jev-preview`, or a pinned ID.
- `questions` (required): map of your id -> question. The id is never shown to the model.
  - `choice`: `instructions` (string/object/array), `criteria` = map option -> description
    (or null). Max 255 options.
  - `noul`: yes/no, optional `criteria.true` / `criteria.false`.
  - `score`: `criteria` = ordered array of 2-10 levels.

**Response**: `model` (versioned ID that answered, log it), `answers` keyed by your ids,
`usage.input_tokens` / `usage.output_tokens`. Choice answer: `type`, `choice` (argmax),
`probabilities` (map option -> float, sums to 1), `confidence` (derived from probabilities;
Kev documents it as `(p_max - 1/K) / (1 - 1/K)`, the formula of TypeSafe's reference adapter).

## 4. For our use

Yes: a `choice` question returns one of N options plus a probability per option.
Mapping for one ASIMOV text item: **one call per item**, `state` = the scenario, four `choice`
questions (ids `q1`..`q4`), each with `criteria` = `{"A": "<option text>", ...}` (5/4/4/4
options). Questions are evaluated in parallel and in isolation against the same state, so
packing four into one call does not let answers see each other (Kev: probabilities within
4e-6 together vs separate). Notes:
- Jev's docs list "Choice option order" as a known weakness: jev-1.13 leans toward the
  **first** option. Worth a permutation check (Kev has `/permute` built in).
- Probabilities are only relative to the supplied options (no "none of these" unless we add one).
- Jev is **text only**: no image or video input. Only the text benchmark applies.
- Cost estimate: 319 items x ~1-2k input tokens ~ 0.3-0.6M tokens ~ $0.01-0.03
  (**UNVERIFIED**, depends on the wording rule).

## 5. Limits and errors

Jev 1.13: 100K tokens/s and 80 requests/s (docs warn these change without notice);
context 64k tokens per request, 32k for `state` + longest question.
Errors (Jev API reference), JSON body:

| Status | Meaning |
|---|---|
| 401 | Missing or invalid API key |
| 422 | Body failed validation (missing field, malformed question); body names the field |
| 429 | Rate limit; may carry `retry-after`; retry with exponential backoff |
| 529 | Overloaded; retry with backoff |

The Python SDK also defines 400, 403 and 404 exceptions. Its default retry policy is 2
retries on 408, 429 and 5xx, with backoff from 0.5 s to 5 s, honoring `retry-after`, and a
30 s timeout. Kev returns 422 for an over-long state (65,536 tokens + 8,192 per question).
Open-Jev returns 413 (body > 4 MB), 415 (wrong content type), 422 (bad request) and 500
(inference failed).

## 6. Example (copied from docs.typesafe.ai/api)

```json
POST https://api.typesafe.ai/v1/systemone
{"state": "Help! My payouts have been failing for 3 days.",
 "model": "jev-latest",
 "questions": {"department": {"type": "choice",
   "instructions": "Which team should handle this?",
   "criteria": {"billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts"}}}}
```
```json
{"model": "jev-1.13.0",
 "answers": {"department": {"type": "choice", "choice": "billing",
   "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
   "confidence": 0.81}},
 "usage": {"input_tokens": 318, "output_tokens": 34}}
```

## 7. Hardware (local models)

- **Kev-4B**: 24 GB GPU (about 16 GB free; 14.3 GB resident) or a 32 GB Apple Silicon Mac
  (MLX, about 8.4 GB of weights). Kev-0.8B fits a 4 GB GPU or any Apple Silicon Mac.
  CPU works (`--device cpu`, fp32), with no published speed. Python 3.12/3.13 with `uv`.
  Install: `git clone https://github.com/jaredpalmer/kev && cd kev && uv sync --extra serve`.
- **Open-Jev-2B**: NVIDIA GPU (6.9 GiB peak recorded), CPU (much slower), or MPS in fp32.
  The 9B needs one 16 GB card. The 27B needs an H100 80 GB.
  Install: `pip install -e '.[train]'` and `hf download ZefanCai/Open-Jev-2B --revision 0c7aa49...`.
- **Laya**: 421M params (ModernBERT-large), so likely CPU-feasible (**UNVERIFIED**). Its own
  card says the base checkpoints sit below the majority baseline on its typed-decision
  benchmark, and that accuracy comes from fine-tuning.

## 8. Status

`UNKNOWN: Jev needs a paid TypeSafe key (no free tier found); confirm which "OpenJev" PLAN.md means; Kev-4B local on a 32 GB Mac is READY to try.`
