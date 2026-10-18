# video_decision_variants: Jev on the video benchmark, three ways to help a classifier

**Status: EXPLORATORY** (Tyler, 2026-10-18). Not part of the reproduction: `decision_video_v2` (run vid2)
sends the LLM prompt's own words. Each variant here is `decision_video_v2_<name>`, defined in
[`variants.yaml`](variants.yaml) and built by `src/asimov/bench/video.py`; scoring is unchanged.

| Experiment | Variant | Config | What changes |
|---|---|---|---|
| 1 | `defs` | `configs/video_dec_defs.yaml` | Risk and severity options carry **our own** definitions (the paper defines neither) |
| 2 | `injury` | `configs/video_dec_injury.yaml` | A first call asks the most likely injury (paper's Fig. 1 question; the 29 NEISS types of Fig. 11, "No realistic injury possible" first); a second call asks the 3 questions, seeing that pick |
| 3a | (none) | `configs/video_dec_640.yaml` | Same request, frames at 640x360 instead of 498x280 |
| 4 | `score` | `configs/video_dec_score.yaml` | Severity asked as Jev's ordered `score` question (No, Mild, Moderate, Severe Harm) instead of a choice |
| 3b | `1fps` | `configs/video_dec_1fps.yaml` | 8 frames (one per second) at 896x504; first prompt sentence edited to match |

Compare each with Jev in vid2. The injury pick is not scored (the video answer key's injury letters have
no text); report how often "No realistic injury possible" is picked against the 193 safe clips.

## Results (Jev, 287 clips; balanced risk = mean of the two per-class hit rates)

| Run | Variant | Balanced risk | Severity (n = 77) | Timing within 0.5 s | Input tokens / clip |
|---|---|---|---|---|---|
| vid2 | none (baseline) | 63.5% | 35% | 34% | 6,853 |
| vid3 | 1 `defs` | 65.8% | 38% | 33% | 6,993 |
| vid6 | 2 `injury` | 66.8% | 48% | 38% | 13,788 (2 calls) |
| vid4 | 3a 640x360 | 64.5% | 35% | 33% | 10,661 |
| vid5 | 3b `1fps` | 64.3% | 40% | 39% | 10,058 |
| vid7 | 4 `score` | 64.5% | 38% | 32% | 6,818 |

Only injury-first moved severity beyond noise (+13 points; borderline at n = 77). Injury-first picked "No realistic
injury possible" on 116/193 safe clips and 24/94 risky ones, so the 30-option list did not push it to always name
an injury. Vid3-vid6 ran at the same time, so their latency is inflated; vid7 ran alone.
