# Decision-model APIs: ready / blocked (read 2026-10-12)

Written in Session 1 (docs/PLAN.md) by `docs-reader` subagents from public docs. No endpoint was
called and nothing was installed. Anything not confirmed on an official page is marked
**UNVERIFIED** in the files. Session 3 builds from these; re-check anything it depends on.

Machine checked for local models: Apple M1 Pro, **16 GB** unified memory, no CUDA GPU.

| System (PLAN §4) | File | Access | Hardware / cost | Status | Open before Session 3 |
|---|---|---|---|---|---|
| `jev` (Jev via `/v1/systemone`) | [systemone.md](systemone.md) | paid TypeSafe key (`TYPESAFE_API_KEY`); no free tier found | hosted; ~$0.042 / 1M input tokens, output free | **BLOCKED: no key** | get a key; Jev 1.13 is documented as biased toward the first option (consider permuting) |
| `kev4b` (Kev-4B local, same protocol) | [systemone.md](systemone.md) | Apache-2.0 weights on Hugging Face | docs say 24 GB GPU or **32 GB** Mac | **BLOCKED on this Mac (16 GB)** | a bigger machine or a cloud GPU |
| `openjev_think` | [systemone.md](systemone.md) | at least three projects use the name | depends on which | **UNKNOWN** | Noa: which OpenJev is meant |
| `laya` | [systemone.md](systemone.md) | thin docs, auth UNVERIFIED | — | **UNKNOWN** | its model card says base checkpoints score below majority without fine-tuning |
| `luna_decide` / `luna_gen` (gpt-6-luna) | [openai_decisions.md](openai_decisions.md) | existing `OPENAI_API_KEY`; Decisions API is public beta | Decisions $0.10 / 1M input; Responses $0.10 in / $0.50 out | **READY** | `confidence` meaning and probability order; `effort: "none"` on Responses |
| `qwen_gen` / `qwen_logprob` (Qwen3-4B) | [qwen3_4b_local.md](qwen3_4b_local.md) | Apache-2.0, not gated | 8 GB bf16 weights + ~144 KiB/token KV; fits 16 GB tightly, MPS bf16 UNVERIFIED | **READY to try** | logprob mode can't use the verbatim prompt (it asks for JSON first): needs a prompt variant in `experiments/` |
| `nli_zeroshot` (optional) | [nli_zeroshot.md](nli_zeroshot.md) | MIT, free | CPU-feasible, ~870 MB | **READY** | 512-token limit: full prompt doesn't fit; wording rule (PLAN §4.3) decides |
| `semif` (optional) | [semif.md](semif.md) | MIT, local CLI (Qwen3.5-4B, option logits) | MLX/MPS or CPU | **READY to try** | README calls it "formerly OpenJev": may be the same as `openjev_think`, and overlaps with `qwen_logprob` |

Questions for Noa (Part 2 scoping, not decided here): which OpenJev; whether `semif` adds a
contrast beyond `qwen_logprob`; whether to get a TypeSafe key and/or GPU time for Jev and Kev.
