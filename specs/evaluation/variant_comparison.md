---
id: EVAL-004
title: "Variant Comparison"
status: implemented
module: analysis.py
last_synced: 2026-07-18
---

# Variant Comparison

## Summary

Computes a relative improvement score for each activation variant against
the GELU baseline on TabArena results, normalized per dataset so datasets
with different absolute score scales don't dominate the comparison, and
runs a paired t-test across datasets to flag whether the improvement is
statistically significant.

## Acceptance Criteria

### load_tabarena_scores()

- [x] AC-1: Reads `results/{activation}/seed_*/benchmark_final/tabarena_exp/nanotabpfn_summary.csv` for every seed of the given activation.
- [x] AC-2: Returns a dict mapping `task_id` to the list of per-seed scores for the requested metric column.
- [x] AC-3: Skips seed directories with no `nanotabpfn_summary.csv` and rows with a missing/NaN value for the requested metric.

### aggregate_seed_scores()

- [x] AC-4: Averages the per-seed score list for each `task_id` into a single mean score.

### compute_relative_improvement()

- [x] AC-5: For `higher_is_better=True` (ROC-AUC), improvement is `(variant - baseline) / |baseline|` per dataset.
- [x] AC-6: For `higher_is_better=False` (log loss), improvement is `(baseline - variant) / |baseline|` per dataset, so a lower variant loss counts as a positive improvement.
- [x] AC-7: Only computes improvement for `task_id`s present in both baseline and variant; datasets present in only one are dropped.
- [x] AC-8: Normalizing by the baseline score means a fixed absolute gap counts for more on datasets with a smaller baseline score, preventing large-scale datasets from dominating the aggregate.

### paired_significance_test()

- [x] AC-9: Pairs baseline and variant scores by `task_id` (one pair per dataset, each side already averaged across seeds).
- [x] AC-10: Runs `scipy.stats.ttest_rel` on the paired vectors and returns `t_statistic`, `p_value`, `significant` (`p_value < 0.05`), and `n_datasets`.
- [x] AC-11: Returns `significant=False` and NaN statistics when fewer than 2 common datasets are available (a t-test is undefined below that).

### compare_variant_to_baseline()

- [x] AC-12: Orchestrates load → seed-aggregate → relative improvement → significance test for one `(baseline, variant, metric)` triple.
- [x] AC-13: Returns `mean_relative_improvement_pct` as the mean of per-dataset relative improvements, expressed as a percentage.

### scripts/compare_variants.py

- [x] AC-14: Discovers all activation subdirectories under `--results_dir` other than `--baseline`.
- [x] AC-15: Runs the comparison for both `roc_auc` and `log_loss` for every discovered variant.
- [x] AC-16: Prints the resulting table and saves it to `--output_csv` (default `{results_dir}/variant_comparison.csv`).

## Notes

- ROC-AUC is bounded to `[0, 1]`, so it is the more robust metric for the significance test; log loss is unbounded and a few overconfident wrong predictions on a hard dataset can dominate its mean, so it is reported as a secondary/diagnostic signal alongside ROC-AUC rather than the primary decision metric.
- Pairing is done per-dataset after averaging each activation's score across its 10 seeds, not per-(dataset, seed) — seed indices between two independent training runs aren't causally linked, so pairing by seed index would conflate seed variance with the variant effect. Pairing by dataset isolates exactly the effect the task asked about: that absolute differences vary by TabArena dataset.
- `t_statistic` sign for log loss should be read as: positive means the variant's raw log loss is *higher* (worse) than baseline's, since `ttest_rel` is called on raw (variant, baseline) values, not the improvement-adjusted ones.
