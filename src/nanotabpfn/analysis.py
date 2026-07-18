"""Compare activation variants against a baseline on TabArena results.

Computes per-dataset relative improvement (to normalize away the fact that
absolute score differences vary widely across TabArena datasets) and a
paired t-test across datasets to assess statistical significance.
"""

from pathlib import Path

import pandas as pd
from scipy import stats

METRICS = {
    "roc_auc": True,  # higher is better
    "log_loss": False,  # lower is better
}


def load_tabarena_scores(results_dir: Path | str, activation: str, metric: str) -> dict[str, list[float]]:
    """Load per-dataset scores for one activation across all its seed runs.

    Args:
        results_dir: Base results directory (e.g. `results/`).
        activation: Activation subdirectory name (e.g. `"gelu"`, `"swiglu"`).
        metric: Column to read from `nanotabpfn_summary.csv` (`"roc_auc"` or `"log_loss"`).

    Returns:
        Mapping of `task_id` to the list of per-seed scores for that dataset.
    """
    activation_dir = Path(results_dir) / activation
    scores: dict[str, list[float]] = {}
    for seed_dir in sorted(activation_dir.glob("seed_*")):
        summary_path = seed_dir / "tabarena_exp" / "nanotabpfn_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path, index_col="task_id")
        for task_id, row in df.iterrows():
            if metric not in row or bool(pd.isna(row[metric])):
                continue
            scores.setdefault(str(task_id), []).append(float(row[metric]))
    return scores


def aggregate_seed_scores(scores: dict[str, list[float]]) -> dict[str, float]:
    """Average per-dataset scores across seeds.

    Args:
        scores: Mapping of `task_id` to a list of per-seed scores.

    Returns:
        Mapping of `task_id` to the mean score across seeds.
    """
    return {task_id: sum(values) / len(values) for task_id, values in scores.items()}


def compute_relative_improvement(
    baseline: dict[str, float], variant: dict[str, float], higher_is_better: bool
) -> dict[str, float]:
    """Compute per-dataset relative improvement of a variant over the baseline.

    Normalizes by the baseline score so that datasets with very different
    absolute score scales contribute comparably, instead of raw absolute
    differences being dominated by whichever datasets happen to have the
    largest scale.

    Args:
        baseline: Mapping of `task_id` to baseline score.
        variant: Mapping of `task_id` to variant score.
        higher_is_better: Whether a higher raw score is an improvement
            (`True` for ROC-AUC, `False` for log loss).

    Returns:
        Mapping of `task_id` to relative improvement as a fraction (e.g.
        `0.05` means +5%), restricted to datasets present in both inputs.
    """
    common_tasks = sorted(set(baseline) & set(variant))
    improvements = {}
    for task_id in common_tasks:
        base_score = baseline[task_id]
        
        if base_score == 0.0:
                improvements[task_id] = float('nan')
                continue
                
        variant_score = variant[task_id]
        diff = (variant_score - base_score) if higher_is_better else (base_score - variant_score)
        improvements[task_id] = diff / abs(base_score)
    return improvements


def paired_significance_test(baseline: dict[str, float], variant: dict[str, float]) -> dict:
    """Run a paired t-test between baseline and variant scores across datasets.

    Samples are paired by `task_id` (each dataset contributes one baseline
    score and one variant score, typically already averaged across seeds),
    which isolates the variant-vs-baseline effect from dataset-to-dataset
    variance.

    Args:
        baseline: Mapping of `task_id` to baseline score.
        variant: Mapping of `task_id` to variant score.

    Returns:
        Dict with `t_statistic`, `p_value`, `significant` (p < 0.05), and
        `n_datasets`.
    """
    common_tasks = sorted(set(baseline) & set(variant))
    baseline_values = [baseline[t] for t in common_tasks]
    variant_values = [variant[t] for t in common_tasks]

    if len(common_tasks) < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "n_datasets": len(common_tasks),
        }

    result = stats.ttest_rel(variant_values, baseline_values)
    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
        "n_datasets": len(common_tasks),
    }


def compare_variant_to_baseline(
    results_dir: Path | str, baseline_activation: str, variant_activation: str, metric: str
) -> dict:
    """Compare one activation variant to the baseline for a single metric.

    Args:
        results_dir: Base results directory (e.g. `results/`).
        baseline_activation: Baseline activation name (e.g. `"gelu"`).
        variant_activation: Variant activation name (e.g. `"swiglu"`).
        metric: `"roc_auc"` or `"log_loss"`.

    Returns:
        Dict with `variant`, `metric`, `mean_relative_improvement_pct`,
        `t_statistic`, `p_value`, `significant`, and `n_datasets`.
    """
    higher_is_better = METRICS[metric]

    baseline_scores = aggregate_seed_scores(load_tabarena_scores(results_dir, baseline_activation, metric))
    variant_scores = aggregate_seed_scores(load_tabarena_scores(results_dir, variant_activation, metric))

    improvements = compute_relative_improvement(baseline_scores, variant_scores, higher_is_better)
    mean_improvement_pct = 100 * sum(improvements.values()) / len(improvements) if improvements else float("nan")

    significance = paired_significance_test(baseline_scores, variant_scores)

    return {
        "variant": variant_activation,
        "metric": metric,
        "mean_relative_improvement_pct": mean_improvement_pct,
        **significance,
    }
