# `/v1/systemone` (Jev, OpenJev, Laya, Kev)

Read 2026-10-12. Facts below come from the official TypeSafe docs or each project's own
README / model card unless marked **UNVERIFIED**. No endpoint was called.

**Corrected 2026-10-12 (Session 3):** the first version described Zefan-Cai/Open-Jev and local
Kev/Laya. Noa's Part 2 decisions (WORKLOG 2026-10-12, "Part 2 approved") are: OpenJev =
**razorback16/openjev hosted on Codiv**, Laya **hosted on Codiv**, Kev-4B **self-deployed on
Modal**. Sections 1-3 are rewritten for those; Zefan-Cai/Open-Jev is no longer used.

## 1. What it is

`POST /v1/systemone` is TypeSafe's "System One" API: you send a `state` (text or JSON) and a
map of typed questions, and you get back typed answers with probabilities. No text is generated.
- **Jev**: TypeSafe's closed hosted model (current `jev-1.13.0`).
- **OpenJev** (https://github.com/razorback16/openjev): `openjev-latest` (alias `openjev-0.1`),
  DiffusionGemma 26B-A4B (26B total, 4B active; base `nvidia/diffusiongemma-26B-A4B-it-NVFP4`,
  Apache-2.0), answers read from token probabilities. Adds an optional request field
  `think` (0-4096 tokens, "a hard cap"; "Give multi-step problems 512 or more"); not part of
  Jev's API; `usage.output_tokens` is 0 unless `think` is set. Default not stated (implied 0).
- **Laya** (`laya-1.0`): 421M ModernBERT-large encoder, text only, served by Codiv.
- **Kev** (https://github.com/jaredpalmer/kev): Qwen3.5-4B-Base + LoRA + pointer head.
- **Codiv** (https://codiv.ai) is "an inference platform for open System One models" with "the
  same wire API as TypeSafe's Jev". It lists `openjev-latest`, `laya-1.0`, `verdict-1.4`,
  `clm-v0.1`, `jevk5-0.3`, `clef-flash-1.0`. It also accepts the aliases `jev-latest` and
  `jev-preview` (OpenJev README), so the base URL, not the model name, says which service answered.

## 2. Access

| System | Access | Auth (official) | Our env vars | Price |
|---|---|---|---|---|
| Jev | Paid API, key from https://console.typesafe.ai/keys | `Authorization: Bearer <key>`; SDK env var `TYPESAFE_API_KEY` | `TYPESAFE_API_KEY` | $0.042 per M input tokens, output free |
| OpenJev, Laya | Codiv account, keys look like `sk-codiv-...` | `Authorization: Bearer`; Codiv's docs set `TYPESAFE_API_KEY` + `TYPESAFE_BASE_URL=https://api.codiv.ai` | **`CODIV_API_KEY`** (our name, see below) | free tier: 100M input tokens per account, no card; 60 requests/min per key |
| Kev-4B | Self-deployed on Modal (Apache-2.0 weights, HF `jaredpalmer/kev-4b`, tag `v1.0`) | `Authorization: Bearer <KEV_API_KEY>` when `KEV_API_KEY` is set at deploy | `KEV_API_KEY`; the `modal.run` URL goes in `configs/part2_kev.yaml` (`base_url`) | Modal GPU time (L40S) |

**Key names:** Codiv's official setup reuses `TYPESAFE_API_KEY`, which would collide with the Jev
key in one `.env`. Our client reads `CODIV_API_KEY` for Codiv instead (config `key_env`).

## 3. Endpoint / interface

- Jev: `POST https://api.typesafe.ai/v1/systemone`, `Content-Type: application/json`.
  `GET /v1/models` lists the names your account can send. No `think` field documented.
- OpenJev / Laya: `POST https://api.codiv.ai/v1/systemone`, models `openjev-latest`, `laya-1.0`.
  `think` is not supported with `images` or `sequential` (we use neither).
- Kev on Modal (Kev README): `pip install modal && modal setup`, then
  `curl -LO https://raw.githubusercontent.com/jaredpalmer/kev/main/skills/kev-deploy/scripts/kev_serve.py`
  and `KEV_API_KEY=$(openssl rand -hex 24) modal deploy kev_serve.py`. Serves Kev-4B on an L40S
  at `https://<your-workspace>--kev-api.modal.run/v1/systemone`, model `kev-latest`; pin weights
  with `jaredpalmer/kev-4b@v1.0` (revision `139fdd94`). Scales to zero when idle: the first
  request after idle waits about 35 s (exclude from latency, warm up first). Extra endpoints:
  `/v1/systemone/permute` (one Choice over 1-64 option orders, default 6) and
  `/v1/systemone/separate`. Kev's response also has top-level `latency_ms`.

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
- **Correction (2026-10-18):** Jev takes images (experimental, https://docs.typesafe.ai/images):
  `assets` keyed by a reference written into `state`, plus the header `TypeSafe-Danger-Zone-Images: true`.
  16 full-size video frames give `max_tokens_exceeded`; 498x280 frames work (~6,850 input tokens).
  Codiv accepts the same request but the images don't reach OpenJev (WORKLOG 2026-10-18).
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
Codiv: 60 requests/min per key (codiv.ai); its error codes are not documented
(**UNVERIFIED**; we treat them like Jev's).

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

## 7. Hardware

All four are hosted: Jev (TypeSafe) and OpenJev / Laya (Codiv) need nothing local. Kev-4B runs on
a Modal L40S (Kev README); this 16 GB Mac is too small for it locally (docs: 24 GB GPU or 32 GB
Mac). Laya's own card says its base checkpoints sit below the majority baseline on its
typed-decision benchmark, and that accuracy comes from fine-tuning.

## 8. Status

READY, smoked 2026-10-13 (5 items each, 0 errors; WORKLOG): `jev` (`jev-1.13.0`), `openjev` think 0 /
512 (`openjev-0.1`), `laya` (`laya-1.0`), `kev4b` (`kev-latest` = `kev-4b@v1.0`, Modal app `kev`, URL in
`configs/part2_kev.yaml`).
