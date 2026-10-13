# text_letter_only: the paper's text prompt plus one "answer with a letter" sentence

**Status:** available for future runs (Noa, 2026-10-12). Not used by `repro_fig4` or
`repro_fig4_default`, which send the paper's prompt verbatim (`text_verbatim`).

**What it changes:** after the dataset's prompt, unchanged, the request adds a blank line and
the sentence in [`suffix.txt`](suffix.txt). It asks for the option letter only in every
`*_response` field.

**Why:** in the `repro_fig4_default` run, GPT-5 Nano sometimes wrote an option's text instead of
its letter (e.g. `"No realistic injury possible"` for E). The scorer reads those anyway
(`option_text` status, WORKLOG 2026-10-12), but a clearer prompt should make them rarer.

**How to use:** set `prompt: text_letter_only` in a config with its own `run:` name. The cache
key includes the added sentence (`bench/text.py` `prompt_version`), so these replies never mix
with verbatim-prompt replies. Comparing a run with it against the verbatim run on the same items
shows whether the sentence changes format slips, or answers. That's a question for H1, not part of
the reproduction.
