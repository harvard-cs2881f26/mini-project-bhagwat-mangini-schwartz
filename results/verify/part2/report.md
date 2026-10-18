# Part 2 independent check (results/verify/part2/verify_part2.py)

Tolerance 0.001 (or half the last printed digit); bootstrap bounds 0.005.

## part2_designs/tables.md

- checked: 3000, matched: 3000, mismatches: 0

## router_search/tables.md single systems

- checked: 730, matched: 730, mismatches: 0

## router_search/picks.csv

- checked: 44, matched: 44, mismatches: 0

## Notes

- pick parallel/cost/0.1 [OpenJev think 512 full || Qwen3.5-4B generate 2048 in parallel; Qwen3.5-4B generate 2048's answers on the lowest 10% by confidence]: folds rebuilt from WORKLOG (permutation of sorted ids, round-robin, threshold = np.quantile of training confidences, sent if conf < thr); ('sorted ids', '<'): 0.723197; ('sorted ids', '<='): 0.723197; ('summary order', '<'): 0.725862; ('summary order', '<='): 0.725862
- picks.csv rows with self_hosted_pick_share > 0 (WORKLOG says 0 throughout): agreement/cost/0.1=1.0, agreement/cost/8.0=0.04, agreement/cost/16.0=0.04, agreement/cost/32.0=0.04, agreement/cost/inf=0.04, agreement/latency/0.5=0.84, agreement/latency/1.0=0.04, agreement/latency/2.0=0.04, agreement/latency/4.0=0.12, agreement/latency/8.0=0.04, three_stage/cost/0.1=0.36, split_by_question/cost/0.1=1.0, parallel/cost/0.1=1.0, parallel/cost/0.25=0.04, parallel_per_question_mix/cost/0.1=1.0, parallel_per_question_mix/cost/0.25=0.44, parallel_per_question_mix/cost/0.5=0.04
