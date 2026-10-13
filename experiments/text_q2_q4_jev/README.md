# text_q2_q4_jev: the paper's prompt without Q1; Jev's Q1 pick is given

**Status: EXPLORATORY. Wording PROPOSED** (drafted by Claude, 2026-10-14; Noa approved running it under a
$5 cap). Not part of the reproduction: `repro_fig4` sends the paper's prompt verbatim (`text_verbatim`).

**Purpose:** tests the system "Q1 from Jev, Q2-Q4 from the LLM" (design B's pre-registered split), with
the LLM actually told Jev's injury instead of choosing its own. Jev = run `part2_jev` (short wording,
independent questions).

**What it sends:** the dataset's prompt with the Q1 block (question, guidance and options A-E) replaced by
[`template.txt`](template.txt), filled with Jev's Q1 pick as the paper's option text (e.g. `Concussions`), and
Q1's two keys removed from the JSON format line. Everything else (opening, scenario, instruction, Q2-Q4 with
their options, the rest of the JSON format, the closing line) is verbatim, including the paper's references
to "the injury you identified in Q1", which the added sentence points at Jev's pick (`bench/text.py`
`without_q1`; spec in [`variant.yaml`](variant.yaml)). Nothing from the answer key and no item id is sent
(`tests/test_prompt_hints.py`, `tests/test_leakage.py`).

**Scoring:** the reply has no Q1, so this run's own summary scores Q1 as unanswered and every reply as a
parse failure. That is expected, not a bug. The composed system (Q1 = Jev's saved answer, Q2-Q4 = this
run's) is scored by `scripts/prompt_experiments.py`.

**Runs:** `configs/part2_gpt5_q2q4_jev.yaml`, `configs/part2_gpt5mini_q2q4_jev.yaml`.

**Alternatives considered:** omitting Q1 entirely without naming an injury (Q2-Q4 would then refer to an
injury that was never identified); keeping Q1 and asking the LLM to copy Jev's answer.
