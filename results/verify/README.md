# Independent check of reported numbers

`verify.py` recomputes the numbers for `repro_fig4` (text) and `repro_fig5` (video) from the
saved replies and the answer keys. It does this with its own code, written from the rules in
WORKLOG.md and CLAUDE.md, not from `src/asimov/stats.py`, `src/asimov/collect/`, the scorers or
the plot script. A check that reuses the code it checks catches nothing.

## What it checks

For each (run, model), using only the replies under that model's `summary.json` `key`:

- **text:** accuracy on the four questions, severe-vs-not accuracy and severe recall (latent
  severity and activated risk), and parse-failure rate. Replies are read from the last `{...}`
  (repeated keys keep the first value), with a fallback to loose `"key": "value"` pairs. Near-miss
  keys and lenient letters are handled as well.
- **video:** risk accuracy, `within_0_5_rate` over all scored videos, `timing_mae_s` over the error
  cases, and the number of error cases. Answer-key times are snapped to 0.5 s with ties going
  earlier, and each item falls into one of four timing cases.
- **Wilson 95% CIs** against `results/<run>/README.md`.
- **Paired large-minus-small gaps** (seeded bootstrap, 10,000 resamples, percentile), exact
  McNemar p and Holm-adjusted p (16 tests) against `results/repro_fig4/paper_check.md`. The
  "actions harder" table is checked too.
- **Point values** against `results/index.csv`.
- Our reading of each item against `per_item` in `summary.json`. Every disagreement is listed.

Tolerances:

- **Non-bootstrap values** (point values, Wilson bounds, p-values) must match within `TOL = 0.001`,
  or within the rounding of the printed value.
- **Bootstrap bounds** have Monte Carlo noise that depends on n, so a fixed tolerance doesn't suit
  them. Each bound is recomputed with our own percentile bootstrap under `BOOT_SEEDS` 0–19, with
  `BOOT_RESAMPLES` = 10,000 per seed. A reported bound matches if it lies inside
  [min − `BOOT_MARGIN`, max + `BOOT_MARGIN`] of those 20 values, where `BOOT_MARGIN` = 0.002.

## Run

```bash
python -I results/verify/verify.py --data /path/to/data   # main checkout's data/ when in a worktree
```

Exits 1 if anything is outside tolerance.
