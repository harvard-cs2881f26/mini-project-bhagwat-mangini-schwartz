# text_jev_hints: the paper's text prompt plus Jev's 4 picks as hints

**Status: EXPLORATORY. Wording DECIDED** (Noa, 2026-10-14: confirmed as proposed by Claude; drafted by Claude, 2026-10-14; Noa approved running it under a
$5 cap). Not part of the reproduction: `repro_fig4` sends the paper's prompt verbatim (`text_verbatim`).

**Purpose:** does an LLM do better when it sees a cheap decision model's answers first? Jev (run
`part2_jev`, short wording, independent questions) is the pre-registered first model of Part 2. This
variant gives the LLM all 4 of Jev's picks for the item and lets it answer every question itself.

**What it sends:** the dataset's prompt, unchanged, then a blank line and [`template.txt`](template.txt),
filled per item with Jev's picks as `<letter>. <the paper's option text>` (e.g. `Q1: D. Concussions`). The
picks are read from Jev's saved replies under its summary's cache key, with the text scorer's own letter
reading (`bench/text.py`, hint variants; spec in [`variant.yaml`](variant.yaml)). Nothing from the answer
key and no item id is sent (`tests/test_prompt_hints.py`, `tests/test_leakage.py`).

**Scoring:** unchanged; the reply has the paper's JSON format, so all 4 questions are scored as usual.

**Runs:** `configs/part2_gpt5_jev_hints.yaml`, `configs/part2_gpt5mini_jev_hints.yaml`. Analysis:
`scripts/prompt_experiments.py` (WORKLOG 2026-10-14, "Part 2 prompt experiments").

**Alternatives considered:** giving Jev's probabilities as well as its picks; not saying the picks may be wrong.
