# Predeclared thinking-settings experiment

Candidate settings were fixed before the full runs from the user-provided figure data. Hidden values may be from older drafts. No scorer change or outcome-based parameter search was performed. Standard repo thinking=min/default remains; native reasoning_effort / thinking_budget overrides are explicit per model and enter cache keys.

| Model / candidate | Violations / parsed | Our rate | Paper reference | Gap (points) |
|---|---:|---:|---:|---:|
| GPT-5 Mini (high) | 87/164 | 53.0% | 46.9% | +6.1 |
| GPT-5 (medium candidate) | 81/164 | 49.4% | 40.6% | +8.8 |
| Claude Sonnet 4.5 (1024)* | 53/161 | 32.9% | 50.0% | -17.1 |
| Claude Opus 4.5 (4096)* | 61/163 | 37.4% | 38.6% | -1.2 |

All runs returned 164 responses; schema failures remain unknown and empty selections count as nonviolations. Exact Sonnet 4 and Opus 4.1 returned 404 during one-item preflight, so Claude 4.5 results are explicitly substitutes. Mini high compares to the same-role default/medium run. GPT-5 and Claude comparisons to previous baselines change both message roles and thinking. Claude total output cap is thinking budget + the existing 2048 answer-token allowance; OpenAI remains 8192. One stochastic run per setting does not establish author-equivalent reproduction.

Raw responses and per-model summaries remain in `../img4_repro_fig6_candidates/` and `../img5_repro_fig6_candidate_successors/`; manifests record settings, usage, source hashes and paired changes. PNG/PDF/CSV/JSON comparisons are here.
