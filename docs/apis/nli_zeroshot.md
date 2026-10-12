# nli_zeroshot: NLI zero-shot classification (Hugging Face pipeline)

## 1. What it is

A local encoder classifier fine-tuned on natural-language inference (NLI), used through the
`transformers` `zero-shot-classification` pipeline. Each (text, candidate label) pair becomes a
premise/hypothesis pair, `hypothesis_template.format(label)`. The *entailment* logit is the score
for that label. No generation and no API key; one forward pass per (text, label) pair.
Sources, read 2026-10-12:
- Pipeline docs: https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
- Pipeline source: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py
- Checkpoints: https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 ,
  https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0 ,
  https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0 (model cards, `config.json` and `tokenizer_config.json`)

## 2. Access

Public, ungated weights; free. No auth. Licenses: deberta-v3-large-zeroshot-v2.0 MIT,
ModernBERT-large-zeroshot-v2.0 Apache-2.0 (from the model cards). The bge-m3 license was not checked.

**Checkpoint choice** (the owner decides; these are the facts):

| checkpoint | params / size | max positions | card's evidence |
|---|---|---|---|
| `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | ~435M; 870 MB fp16 safetensors | **512** (`model_max_length: 512`) | best mean f1_macro on the card's 28-task zero-shot table (0.673-0.676 for the large variants vs 0.497 for `facebook/bart-large-mnli`) |
| `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | 792 MB bf16 safetensors | **8192** | the card calls it "slightly worse then DeBERTav3 on average" and "multiple times faster" |
| `MoritzLaurer/bge-m3-zeroshot-v2.0` | XLM-R large | **8194** | 0.59 mean on the same table (`-c` variant) |
| `facebook/bart-large-mnli` | ~400M | 1024 (UNVERIFIED, not read) | the old default; the weakest on that table |

The strongest reasonable default for short premises is `deberta-v3-large-zeroshot-v2.0`, which
leads the author's own benchmark. These are 2024 checkpoints. The author's "newer version" built
for the 8k window, mentioned on the ModernBERT card, was not found on the Hub. Both newer
checkpoints use the labels `{"entailment": 0, "not_entailment": 1}`: two classes, not three.

## 3. Endpoint / interface

```python
from transformers import pipeline
clf = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
out = clf(sequences, candidate_labels, hypothesis_template="This example is {}.", multi_label=False)
```
- `sequences`: `str | list[str]`. The docs say it "will be truncated if the model input is too large".
- `candidate_labels`: `str | list[str]` (a list, or one comma-separated string).
- `hypothesis_template`: default `"This example is {}."`; must contain `{}`.
- `multi_label=False`: softmax of the entailment logits **across all labels**, so the scores sum to 1.
  `True`: each label is scored independently, as softmax(entailment vs contradiction).
- Returns a `dict` (or a list of them) with keys `sequence`, `labels` (**sorted by score, not
  input order**), and `scores` (aligned with `labels`).
- The pipeline finds the entailment index as the first `label2id` key starting with "entail". If
  none matches, it logs a warning and uses -1. With `multi_label=True` it also needs a
  contradiction id. How that resolves on a two-class `not_entailment` model is **UNVERIFIED**.
  Use `multi_label=False`.

## 4. For our use

- **One option per answer, with a probability per option: yes.** With `multi_label=False`,
  `scores` is a distribution over the given options. Map each score back to its letter by label
  text, because the output order is sorted.
- **One pipeline call per question, not per item.** Each question has its own options, so an
  item takes 4 calls, each costing (number of options) forward passes. That is about 17-20 passes
  per item and about 6k for 319 items.
- **Our ~1.3k-token prompt does not fit the DeBERTa or BART models.** Truncation is
  `ONLY_FIRST`, so the premise is cut and the hypothesis kept. Measured on `data/text/items.jsonl`:
  prompts are 6,206-7,073 characters, mean 6,543. The scenario and instruction near the start are
  short (roughly 100-150 tokens, **UNVERIFIED**, not tokenized), so the cut would drop the
  questions and rubric at the end. Two ways to handle it:
  (a) premise = scenario (+ instruction for the questions that need it), and the question stem and
      option text go into the hypothesis template. This fits easily in 512 tokens.
  (b) premise = the full verbatim prompt. This needs an 8k model (ModernBERT-large or bge-m3).
  This is PLAN §4.3's wording rule (full rubric vs short instructions). The owner picks and logs
  the choice in WORKLOG.
- Option text: Q1 injury options are short noun phrases ("Concussions"), which suit a template such
  as "The most likely injury is {}." The Q2 severity options are full rubric sentences, so they
  give long hypotheses. Template wording is an experiment choice; it belongs in `experiments/`.
- NLI models are not trained on counterfactual questions like "the effect of executing the
  instruction". Expect weaker results on Q3/Q4. This is a hypothesis to test, not a documented fact.

## 5. Limits and errors

- No rate limits (runs locally). Max input: see the table. If the input is too long it is
  truncated silently, with no error. Log the token count per item so truncation is visible.
- Missing entailment label: a warning, and the outputs are wrong. It does not raise.
- `ChunkPipeline`: one input expands to N forward passes, so `batch_size` counts pairs, not items
  (pipeline docs, "chunk batching").

## 6. Example (copied from the transformers docs)

```python
>>> oracle = pipeline(model="facebook/bart-large-mnli")
>>> oracle("I have a problem with my iphone that needs to be resolved asap!!",
...        candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"])
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!',
 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'],
 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
```

## 7. Hardware

- Install: `pip install "transformers[sentencepiece]" torch` (the DeBERTa card says
  `transformers[sentencepiece]`). `transformers` and `torch` are **not** in our `.venv` yet.
  ModernBERT needs transformers >= 4.48 (its config says `4.48.0.dev0`; minimum **UNVERIFIED**).
- Memory: weights are about 0.8-0.9 GB in fp16/bf16, so they fit easily on any GPU or Apple MPS.
  **CPU-only is feasible** (the card says "run on both GPUs and CPUs"). Speed on this Mac is
  **UNVERIFIED**. As a guide, the ModernBERT card reports about 1,100 texts/s on an A100 at batch 32
  with short texts. A rough guess for about 6k pairs of up to 512 tokens is minutes on CPU, not hours
  (**UNVERIFIED**).
- Inference cost for H1: $0, 0 output tokens. Latency and parameter count are the cost axes.

## 8. Status

`READY`. Public, free, CPU-feasible, with a per-option probability output. Open decision for
the owner: the premise wording (a) vs (b), which also sets the checkpoint (512-token DeBERTa vs
8k ModernBERT/bge-m3).
