# Presentation outline (draft)

Rough structure only. Each owner writes their own slide content, findings and conclusions.
`[TODO owner]` marks what each owner fills in. Citations marked *(verify)* must be checked
against the source before they go on a slide.

Assumes about 15 minutes and about 14 slides. Adjust once we know the slot length.

---

## 1. Background: why physical safety? (2 slides)

**Slide 1: Title + one-line question**
- "Can AI perceive physical danger, and can we make that cheap enough to run on a robot?"

**Slide 2: The problem**
- Foundation models are moving from chat into robots, smart glasses and home and industrial
  agents. Their mistakes can now cause physical injury, not just bad text.
- Text-safety work (refusals, jailbreaks) doesn't cover this. A request can be harmless in
  words and dangerous in the scene ("hand me the cup" when the cup is boiling).
- Deployment constraint: robots often need on-device, low-latency inference, which pushes
  toward small models. That leads into H1.
- `[TODO owner]` One concrete example item from the dataset as the hook (image + constraint).

## 2. Prior work and why this paper matters (2 slides)

**Slide 3: Safety benchmarks so far**
- Text/LLM safety benchmarks: refusal and harm benchmarks, e.g. HarmBench *(verify)*.
- Code security benchmarks: e.g. Meta's CyberSecEval *(verify)*, which measures whether models
  write insecure code or help with attacks.
- The analogy: those benchmarks made "security in code" measurable. ASIMOV does the same for
  "safety in the physical world".
- Predecessor: ASIMOV v1, Google DeepMind 2025, robot constitutions / semantic safety *(verify)*.

**Slide 4: What ASIMOV-2.0 adds**
- Grounded in real ER injury reports (NEISS) and industrial robot standards, not invented
  scenarios.
- Three input types: text scenarios, video, images with a robot's physical constraints.
- Measures both *perceiving* risk and *acting* on it (intervention timing, constraint violations).

## 3. The paper and our reproduction (4 slides)

**Slide 5: Paper intent + the three benchmarks**
- Table: Injury (319, text, Fig. 4) / Video (287, Fig. 5) / Constraints (164, image, Fig. 6).
- Paper headlines: ~92% latent-risk accuracy and ~74% action-safety accuracy for top models;
  small models drop up to 20 points; video is 27–40 points worse than text for some models;
  constraint violation rates of 38.6–75%.

**Slide 6: Fig. 4 Injury, paper vs ours** `[TODO Person 1]`
- Side by side: paper figure | our bars, same metrics.
- Model substitutions (Sept 2025 lineup vs what's served now).

**Slide 7: Fig. 5 Video, paper vs ours** `[TODO Person 2]`
- Caveat to state: the dataset doesn't ship the video prompt. We rebuilt it from the
  appendix, so this is the least exact reproduction.

**Slide 8: Fig. 6 (+7) Constraints, paper vs ours** `[TODO Person 3]`
- Caveat to state: 43% of safe objects sit inside violating boxes (sushi in the bowl), so the
  overlap rule changes the numbers. Show the scoring rule we used, and the other rule if both
  were run.
- Always show violation rate next to parse-failure and empty-answer rate. A model that
  answers `[]` scores 0% violations.

*(Optional backup slide: methods decisions from WORKLOG.md.)*

## 4. Our hypotheses (1 slide)

**Slide 9: Two hypotheses**
- **H1: Small models.** Can prompting, training or other techniques close the gap to large
  models within an on-device budget? Success is measured on *both* accuracy and cost.
- **H2: Safety classifiers.** Do newer LLM-based safety classifiers (similar to
  `[Jev, confirm name]`) give a better accuracy/cost trade-off than general models here?
- Shared axes for both: accuracy (benchmark metric) vs. cost (latency per item, tokens,
  parameter count, $ per 1k items).

## 5. Results (4–5 slides)

**Slide 10: H1 setup** `[TODO owner]`
- Which small models, which benchmark(s), what budget counts as "on-device".
- Candidate techniques for the owner to choose from, not a plan:
  - Prompting: few-shot, chain-of-thought / thinking on vs off, self-consistency (k samples),
    restating the constraint before answering.
  - Training: SFT or RL on *separate* data (never the eval items), distillation from a large
    model's outputs.
  - Inference tricks: quantization, fewer video frames, a cascade (small model first, large
    model only when unsure).

**Slide 11: H1 results** `[TODO owner]`
- Suggested chart: accuracy vs. cost scatter, one point per (model × technique), with the
  Pareto frontier marked and the paper's large-model results as reference lines.

**Slide 12: H2 setup** `[TODO owner]`
- Which classifiers. Candidates *(verify availability)*: Llama Guard, ShieldGemma,
  gpt-oss-safeguard, `[Jev]`.
- Key design choice: classifiers output safe/unsafe, not multiple-choice letters or points.
  State how that maps onto a benchmark (e.g. Injury: "is executing this instruction unsafe?"
  against the ground-truth effect/activated risk). This is open question 5 in the handoff.

**Slide 13: H2 results** `[TODO owner]`
- Same accuracy-vs-cost axes as H1, so the two can be compared directly.

## 6. Close (1 slide)

**Slide 14: Takeaways + limitations** `[TODO team]`
- What reproduced, what didn't, what H1/H2 showed.
- Limitations: model substitutions, reconstructed video prompt, overlap rule, sample sizes
  (164 items means about ±8 points at 95% confidence on a rate near 50%).

---

## Open questions for the team
1. "Jev": which classifier is this? It's referenced in H2.
2. The original plan had a third experiment, "big models / ensembles". It isn't in these two
   hypotheses. Drop it, or keep it as the large-model baseline in slides 6–8 and 11?
3. H1 merges the original Exp 1 (prompting) and Exp 2 (training). One owner or two?
4. Presentation length and format (slides tool, live demo?).
