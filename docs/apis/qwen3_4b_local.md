# Qwen3-4B, local (Hugging Face transformers)

## 1. What it is

Qwen3-4B is Alibaba Qwen's 4.0B-parameter dense causal LM (3.6B non-embedding, 36 layers, GQA 32 Q /
8 KV heads, 32,768-token native context), post-trained with a switchable thinking / non-thinking mode.
We run it locally with `transformers` in two modes: (a) generation with thinking off, (b) multiple-choice
scoring from the next-token logits of the letters A-E. Sources, read 2026-10-12: model card
https://huggingface.co/Qwen/Qwen3-4B (repo commit `1cfa9a7208912126459214e8b04321603b3df60c`, last
modified 2025-07-26), plus that repo's `config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json` and `model.safetensors.index.json` (fetched, no weights downloaded).

## 2. Access

- Public, **not gated** (`gated: false` from the Hub API). No account or token needed to download.
- License: **Apache-2.0** (model card front matter and the repo's `LICENSE` file). Commercial use OK.
- Cost: free; only local compute. Weights are ~8.05 GB to download (3 safetensors shards: 3.96 GB +
  3.99 GB + 0.10 GB).
- Pin the revision in code: `from_pretrained("Qwen/Qwen3-4B", revision="1cfa9a72...")` so reruns use
  the same weights.

## 3. Endpoint / interface

Library calls, no HTTP. From the model card quickstart:

- `AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")`
- `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="auto", device_map="auto")`.
  The card uses `torch_dtype`; in transformers 5.x this argument is named `dtype` (**UNVERIFIED**
  for the exact version where `torch_dtype` stopped working; pass `dtype=torch.bfloat16` and check).
- `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`
- **Thinking off (hard switch):** `enable_thinking=False`. The chat template (read from
  `tokenizer_config.json`) then appends `<|im_start|>assistant\n<think>\n\n</think>\n\n`, i.e. an empty,
  already-closed think block, so the model goes straight to the answer. The `/think` and `/no_think`
  soft switches are ignored when `enable_thinking=False` (model card). Default is `True`.
- Generation: `model.generate(**inputs, max_new_tokens=N)`; decode `output_ids[len(prompt):]` with
  `skip_special_tokens=True`.
- Logprob mode: one forward pass `model(**inputs).logits[0, -1]`, take the logits at the five letter
  ids, softmax over just those five.

Special token ids (tokenizer files): `<|endoftext|>` 151643 (pad), `<|im_start|>` 151644,
`<|im_end|>` 151645 (eos), `<think>` 151667, `</think>` 151668. `add_bos_token: False`.

**Option-letter tokens** (byte-level BPE, read from `tokenizer.json` vocab, confirmed):

| letter | no space (`A`) | leading space (`ĠA` = " A") |
|---|---|---|
| A | 32 | 362 |
| B | 33 | 425 |
| C | 34 | 356 |
| D | 35 | 422 |
| E | 36 | 468 |

All ten are single tokens. Because the thinking-off prompt ends in `\n\n`, the first answer token
has no leading space, so the bare ids 32-36 are the ones to read. Still check at runtime with
`tokenizer.encode("A", add_special_tokens=False) == [32]`. Note `(A` (4346) and `"A` (29133) are also
single merged tokens, so if the prompt makes the model open with `(` or `"`, the letter never appears
as its own next token. See §4.

## 4. For our use

- **Generation (a):** yes, any prompt. One call per item (all 4 questions in one reply) reproduces the
  paper setup; parse with `bench/parse.py`.
- **Logprob (b):** returns a probability per option, but only for the *next* token. The paper's text
  prompt asks for a JSON object, so the first answer token would be `{`, not a letter. For scoring you
  need one forward pass **per question**, with the prompt ending right where the letter goes (e.g. the
  assistant turn pre-filled up to `"answer": "` or a plain "Answer:" format). That makes the prompt
  differ from the paper's verbatim prompt; it belongs in `experiments/` and is a methods choice for the
  owner. So: 4 forward passes per item (one per question); questions with 4 options softmax over A-D,
  5 options over A-E.
- With thinking off, prefilling after `</think>\n\n` is plain string concatenation before tokenizing
  (`tokenize=False`, then append, then tokenize). **UNVERIFIED** whether a JSON prefill shifts the letter
  to the ` A` (space) variant; check the top-1 decoded token on a few items.
- PLAN Step C sanity check: logprob argmax should agree with thinking-off generation on most of 20 items.

## 5. Limits and errors

- Context: `max_position_embeddings` 40,960; native 32,768 (YaRN to 131,072, not needed for us; the card
  says static YaRN can hurt short inputs, so leave `rope_scaling` null).
- `transformers<4.51.0` fails with `KeyError: 'qwen3'` (model card).
- No rate limits. Failure modes are local: out-of-memory, MPS ops not implemented (set
  `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU, **UNVERIFIED** that Qwen3 needs it).
- Endless repetition can happen; card suggests `presence_penalty` (not a plain transformers
  `generate` arg; `repetition_penalty` is the nearest one, **UNVERIFIED** equivalent).

**Recommended sampling (model card, Best Practices):**
- Thinking off: `temperature=0.7, top_p=0.8, top_k=20, min_p=0`.
- Thinking on: `temperature=0.6, top_p=0.95, top_k=20, min_p=0`; card says **do not use greedy** there.
- `generation_config.json` ships `do_sample=True, temperature=0.6, top_p=0.95, top_k=20` (the
  thinking-mode values), so these apply unless overridden. For thinking-off we must pass 0.7/0.8
  explicitly. Greedy (`do_sample=False`) with thinking off is not forbidden by the card but not
  recommended either; for a deterministic, cacheable baseline it's a choice to log in WORKLOG.
- Logprob mode involves no sampling.

## 6. Example

Adapted from the model card quickstart (card code, our two modes; the logprob part is **UNVERIFIED**,
not run):

```python
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="auto")
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok([text], return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
# logprob: prompt ends where the letter goes
logits = model(**tok([text + prefill], return_tensors="pt").to(model.device)).logits[0, -1]
probs = torch.softmax(logits[[32, 33, 34, 35, 36]].float(), dim=-1)  # A-E
```

## 7. Hardware

- **Weights:** 8,044,936,192 bytes in bf16 (`model.safetensors.index.json`), i.e. ~8.0 GB; embeddings
  are tied (`tie_word_embeddings: true`). fp32 would be ~16 GB.
- **KV cache** (computed from config: 36 layers x 8 KV heads x 128 dim x K,V x 2 bytes): ~144 KiB per
  token in bf16, so ~0.3 GB for a 2k-token prompt+reply. Budget ~9-10 GB total for batch 1
  (**UNVERIFIED**, not measured).
- **NVIDIA GPU:** a 12 GB+ card fits bf16 comfortably (**UNVERIFIED**, arithmetic only).
- **Apple Silicon (MPS):** Noa's machine is an M1 Pro with 16 GB unified memory. bf16 on MPS needs a
  recent macOS/PyTorch (**UNVERIFIED**: bf16 MPS support was added around macOS 14 / torch 2.x; if it
  errors, use `dtype=torch.float16`, which halves to the same ~8 GB but may overflow,
  **UNVERIFIED** for Qwen3). 8 GB of weights on 16 GB shared memory should load, but expect memory
  pressure with other apps open. Use `device_map="mps"` or `.to("mps")`.
- **CPU-only:** feasible for 319 items in logprob mode (one forward pass per question); generation will
  be slow (**UNVERIFIED** speed, likely a few tokens/s). Load in bf16 (~8 GB); fp32 (~16 GB) won't fit
  on a 16 GB machine.
- **Install** (heavy deps go in `requirements-local.txt`, per PLAN Step C):
  `pip install "transformers>=4.51.0" torch accelerate` (accelerate needed for `device_map="auto"`).
  Latest on PyPI on 2026-10-12: transformers 5.19.0, torch 2.14.1, accelerate 1.15.0. None of them
  installed in `.venv` yet. Pin the versions you actually install.

## 8. Status

`READY` (public Apache-2.0 weights, tokenizer facts confirmed from repo files; nothing installed or run
yet, so memory, MPS dtype and speed are unmeasured).
