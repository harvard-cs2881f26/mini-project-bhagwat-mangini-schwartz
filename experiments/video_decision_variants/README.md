# video_decision_variants: Jev on the video benchmark, three ways to help a classifier

**Status: EXPLORATORY** (Tyler, 2026-10-18). Not part of the reproduction: `decision_video_v2` (run vid2)
sends the LLM prompt's own words. Each variant here is `decision_video_v2_<name>`, defined in
[`variants.yaml`](variants.yaml) and built by `src/asimov/bench/video.py`; scoring is unchanged.

| Experiment | Variant | Config | What changes |
|---|---|---|---|
| 1 | `defs` | `configs/video_dec_defs.yaml` | Risk and severity options carry **our own** definitions (the paper defines neither) |
| 2 | `injury` | `configs/video_dec_injury.yaml` | A first call asks the most likely injury (paper's Fig. 1 question; the 29 NEISS types of Fig. 11, "No realistic injury possible" first); a second call asks the 3 questions, seeing that pick |
| 3a | (none) | `configs/video_dec_640.yaml` | Same request, frames at 640x360 instead of 498x280 |
| 3b | `1fps` | `configs/video_dec_1fps.yaml` | 8 frames (one per second) at 896x504; first prompt sentence edited to match |

Compare each with Jev in vid2. The injury pick is not scored (the video answer key's injury letters have
no text); report how often "No realistic injury possible" is picked against the 193 safe clips.
