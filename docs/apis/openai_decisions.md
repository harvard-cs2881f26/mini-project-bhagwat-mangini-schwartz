# OpenAI Decisions API (gpt-6-luna): `luna_decide`, plus `luna_gen` notes

## 1. What it is
A dedicated endpoint that turns text/images into typed answers instead of free text. Three
question types: `predicate` (P(statement true)), `choice` (pick one of N supplied values, with a
probability per value), `score` (rate on ordered levels). Public beta since 2026-10-06; GA
"expected in the coming weeks". `gpt-6-luna` is the only supported model. Changelog says it is
"10x faster than the Responses API".
Sources (read 2026-10-12): https://developers.openai.com/api/docs/guides/decisions ,
https://developers.openai.com/api/docs/changelog , https://developers.openai.com/api/docs/models/gpt-6-luna ,
https://developers.openai.com/api/docs/pricing , https://developers.openai.com/api/docs/guides/reasoning .
Field names below were also checked against the installed SDK's generated types
(`openai==3.29.0`, `.venv/.../openai/types/decision*.py`, "generated from our OpenAPI spec").
The API reference page (`/api/reference/resources/decisions...`) returned 404 when fetched.

## 2. Access
- Public beta, open to all developers; no waitlist mentioned. Auth: standard OpenAI key,
  `Authorization: Bearer $OPENAI_API_KEY` (SDK reads `OPENAI_API_KEY`).
- Python SDK: guide says `openai>=3.26.0`; our venv has 3.29.0 and `client.decisions` exists.
- **Pricing, Decisions API (guide):** $0.10 / 1M input tokens; input only, "no cache-read,
  cache-write, or output-token charges". Regional-processing premiums and long-context input
  multipliers apply (multiplier value for Decisions not stated: **UNVERIFIED**, pricing page
  doesn't list Decisions at all). Batch/Flex for Decisions: not mentioned, assume none (**UNVERIFIED**).
- **Pricing, gpt-6-luna on Responses (`luna_gen`)**, pricing page, per 1M tokens,
  short context: input $0.10, cached $0.01, cache write $0.125, output $0.50; long context:
  $0.20 / $0.02 / $0.25 / $0.75. Batch and Flex are half; "Fast" (priority) is double.
  Reasoning tokens bill as output (standard OpenAI behaviour; **UNVERIFIED** for luna specifically).
- Caching on Decisions: forum users say none yet (not official). `usage` still has
  `cached_tokens` / `cache_write_tokens` fields.

## 3. Endpoint / interface
`POST https://api.openai.com/v1/decisions` ; SDK `client.decisions.create(...)` (async: same on `AsyncOpenAI`).

Request (`DecisionCreateParams`):
| field | req | notes |
|---|---|---|
| `model` | yes | `"gpt-6-luna"` only |
| `input` | yes | string, or list of user messages with parts `{"type":"input_text","text":...}` / `{"type":"input_image","image_url":"data:...;base64,...","detail":"low"\|"high"\|"auto"\|"original"}`. Inline data URLs only; max 128 images per request; no external URLs, files, audio, tools |
| `questions` | yes | list; each has `type`, `instructions` (req), `name` (optional, echoed back; guide says must be unique) |
| `safety_identifier` | no | opaque end-user id |

Question shapes:
- `choice`: `choices: [{"value": str\|bool, "description": str (optional)}]`, **2 to 255 choices**, unique.
  (There is no field called `options`; it is `choices`. Per-question text is `instructions`;
  there is no top-level `instructions`.)
- `predicate`: just `instructions`.
- `score`: `levels: [{"label": str, "description": str?}]`, lowest to highest.

No `reasoning`, `temperature`, `seed` or `max_output_tokens` parameter exists in the SDK params.

Response (`Decision`): `answers: [...]`, `model: str`, `usage`.
- choice answer: `{"type":"choice","name":..,"choice": <value>,"confidence": float,
  "probabilities":[{"value": <value>,"probability": float}, ...]}`
- predicate: `{"type":"predicate","name":..,"probability": float}`
- score: `{"type":"score","name":..,"score": float,"confidence": float,
  "probabilities":[{"value": int,"label": str,"probability": float}]}` (score = prob-weighted level index from 0)
- refusal: `{"type":"refusal","name":..}`; per question, other questions still answered.
- `usage`: `input_tokens`, `input_tokens_details{cached_tokens, cache_write_tokens}`,
  `output_tokens`, `output_tokens_details{reasoning_tokens}`, `total_tokens`.
- Unknown, **UNVERIFIED**: whether `probabilities` always sums to exactly 1, whether
  `choice` is always the argmax, what `confidence` is (max prob? something else?), whether answer
  order follows question order (match by `name`), whether probabilities come back in choice order.

## 4. For our use
- Yes: a `choice` question returns one chosen value plus a probability per option.
- **One call per item**: `input` = scenario text; 4 `choice` questions named e.g. `q1..q4`,
  `choices` = letters `"A".."E"` as `value` with the option text as `description` (5/4/4/4).
  Map back by `value`, not position. Wording (rubric text vs short instruction) goes in each
  question's `instructions`, per PLAN §4 wording rule.
- Cost is tiny: input-only at $0.10/1M. 319 items x ~1-2k tokens is well under $0.10 (estimate).
- Refusals arrive per question as `type:"refusal"`; record as wrong per PLAN.
- Forum report (user "platypus", not official): `choice` probabilities looked overconfident
  vs `predicate` on a biased-coin test. Relevant to Brier/ECE; treat as anecdote.
- **`luna_gen` minimal reasoning:** Responses API `reasoning={"effort": ...}`. The gpt-6-luna
  model page lists efforts `none`, `low`, `medium` (default), `high`, `xhigh`, `max`;
  **`minimal` is not listed**, so use `reasoning={"effort": "none"}`. Whether `"minimal"` is
  rejected (400) or mapped for luna: **UNVERIFIED**. Decisions has no effort knob; its internal
  reasoning (if any) shows only in `usage.output_tokens_details.reasoning_tokens`.
- The model page lists supported endpoints as Chat Completions, Responses, Batch, and does
  not list Decisions (page probably predates it).

## 5. Limits and errors
- Rate limits for gpt-6-luna (model page, RPM / TPM): Build 5,000 / 2M; Launch 10,000 / 10M;
  Grow 30,000 / 180M. Whether Decisions shares these or has its own pool: **UNVERIFIED**.
- Context: 1,050,000 tokens (max input 922,000) on the model page; Decisions input limit
  not stated (**UNVERIFIED**). Max questions per request: not stated (**UNVERIFIED**).
- Errors (general error-codes guide, nothing Decisions-specific): 400 bad request
  (`BadRequestError`), 401 bad key (`AuthenticationError`), 403 region (`PermissionDeniedError`),
  429 rate/quota/spend limit (`RateLimitError`; retry with backoff, but quota-exhausted 429s won't
  clear), 500 server (`InternalServerError`, retry), 503 overloaded (honour `Retry-After`).
  Decisions-specific 400 reasons (e.g. <2 choices, duplicate values, non-data-URL image) are
  implied by the schema but their error messages are **UNVERIFIED**.

## 6. Example
Request built from the SDK types, **UNVERIFIED** as a whole (not copied from the docs, not run):
```python
from openai import OpenAI
client = OpenAI()
d = client.decisions.create(
    model="gpt-6-luna",
    input="<scenario text>",
    questions=[{"type": "choice", "name": "q1", "instructions": "<question 1 rubric>",
                "choices": [{"value": "A", "description": "<option A>"},
                            {"value": "B", "description": "<option B>"}]}],
)
for a in d.answers:
    if a.type == "choice":
        print(a.name, a.choice, {p.value: p.probability for p in a.probabilities})
    elif a.type == "refusal":
        print(a.name, "refused")
```
Response shape (illustrative, values made up): `{"model":"gpt-6-luna","answers":[{"type":"choice",
"name":"q1","choice":"A","confidence":0.9,"probabilities":[{"value":"A","probability":0.9},
{"value":"B","probability":0.1}]}],"usage":{"input_tokens":..,"output_tokens":..,...}}`.
The guide's own code blocks (curl/Python/TS/Go/Java/Ruby) should be copied into test fixtures
from the page directly in Session 3.

## 7. Hardware
Hosted; n/a.

## 8. Status
`READY` (public beta, SDK installed; check `confidence` semantics and probability ordering on a `--limit 5` smoke, and that `effort: "none"` is accepted for `luna_gen`).
