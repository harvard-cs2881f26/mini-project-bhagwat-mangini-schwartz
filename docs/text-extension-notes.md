# Text extension notes (Part 2)

Working notes for Noa, kept for later. Not decided; nothing here is a hypothesis or a method yet.

## Which questions suit a decision model

Drafted by Claude on 2026-10-12. The "fit" column applies vendor guidance: decision models are
pitched for quick, one-step judgments between a few clearly described options, and their docs
list multi-step reasoning, "what happens after X" questions and ordered scales as weak spots.
The research behind it is in `reports/System One models accuracy cost.md`.
The accuracy column is the range over the 8 LLMs in `repro_fig4` (thinking
`min`), from each `results/repro_fig4/<model>/summary.json`.

| Question | What it asks | Fit for a decision model | Part 1 LLM accuracy (8 models) |
|---|---|---|---|
| Q1 injury type | pick 1 of 5 described injuries from the scenario | **best fit**: one step, distinct options | 0.86–0.93 (Nano 0.56) |
| Q2 latent severity | rate severity on a 4-level scale | medium: ordered scale, and depends on Q1's injury | 0.66–0.77 (Nano 0.53) |
| Q3 effect of instruction | imagine the robot acts; does risk go up or down? | weak: needs reasoning about what happens next | 0.56–0.69 (Nano 0.47) |
| Q4 activated severity | severity after the robot acts | weakest: needs reasoning about what happens next, and depends on Q1's injury | 0.60–0.67 (Nano 0.40) |
