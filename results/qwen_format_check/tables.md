# Qwen format check: txt38_part2_qwen_gen_fewshot (variant) vs txt33_part2_qwen_gen_2048 (baseline)

**EXPLORATORY: prompt variant, not the paper's prompt word for word**

Built by `scripts/qwen_format_check.py --config configs/analysis/qwen_format_check.yaml` from saved replies ($0). 319 items in both runs.

## Parse failures and causes

A failing item can have several causes.

| Run | Parse fail | invalid E | invalid other | cut off | missing key | no JSON |
|---|---|---|---|---|---|---|
| txt33_part2_qwen_gen_2048 (baseline) | 26 (8.2%) | 20 | 0 | 5 | 1 | 0 |
| txt38_part2_qwen_gen_fewshot (variant) | 0 (0.0%) | 0 | 0 | 0 | 0 | 0 |

## Accuracy (Wilson 95% CI)

Strict: unanswered = wrong, all items. Q1-matched: Q2-Q4 on the items whose Q1 matched the key, within each run (the subsets differ between runs).

| Run | Q1 | Q2 | Q3 | Q4 | Q1 answered E (key E) | Q1 matched (n) | Q2 matched | Q3 matched | Q4 matched |
|---|---|---|---|---|---|---|---|---|---|
| txt33_part2_qwen_gen_2048 (baseline) | 77.1% [72.2%, 81.4%] | 61.8% [56.3%, 66.9%] | 59.6% [54.1%, 64.8%] | 58.0% [52.5%, 63.3%] | 13 (5) | 246 | 70.3% [64.3%, 75.7%] | 61.4% [55.2%, 67.2%] | 62.2% [56.0%, 68.0%] |
| txt38_part2_qwen_gen_fewshot (variant) | 70.8% [65.6%, 75.6%] | 63.9% [58.5%, 69.0%] | 64.9% [59.5%, 69.9%] | 54.2% [48.7%, 59.6%] | 28 (5) | 226 | 71.2% [65.0%, 76.7%] | 65.9% [59.5%, 71.8%] | 58.4% [51.9%, 64.6%] |

## Paired comparison (variant - baseline)

Mean of the 4 questions per item: baseline 0.641, variant 0.635, difference -0.006 [-0.038, +0.024] (paired bootstrap, 10,000 resamples, seed 0).

| Question | variant right, baseline wrong | baseline right, variant wrong | McNemar p (exact) |
|---|---|---|---|
| latent_risk | 7 | 27 | 0.000821 |
| latent_severity | 52 | 45 | 0.543 |
| effect | 38 | 21 | 0.0363 |
| activated_risk | 39 | 51 | 0.246 |

## Cost per item

| Run | Output tokens p50 / p95 | Input tokens p50 | compute_s p50 / p95 | Cut off (all replies) |
|---|---|---|---|---|
| txt33_part2_qwen_gen_2048 (baseline) | 434 / 1648 | 1418 | 16.8 / 65.3 | 11 |
| txt38_part2_qwen_gen_fewshot (variant) | 156 / 184 | 2207 | 6.3 / 7.5 | 0 |
