"""Tests for relative improvement scoring and significance testing."""

from nanotabpfn.analysis import (
    aggregate_seed_scores,
    compute_relative_improvement,
    paired_significance_test,
)


def test_aggregate_seed_scores_averages_across_seeds():
    scores = {"task_a": [0.8, 0.9, 1.0], "task_b": [0.5, 0.5]}
    aggregated = aggregate_seed_scores(scores)
    assert aggregated["task_a"] == 0.9
    assert aggregated["task_b"] == 0.5


def test_relative_improvement_higher_is_better():
    baseline = {"task_a": 0.5, "task_b": 0.8}
    variant = {"task_a": 0.6, "task_b": 0.4}
    improvements = compute_relative_improvement(baseline, variant, higher_is_better=True)
    assert improvements["task_a"] == (0.6 - 0.5) / 0.5
    assert improvements["task_b"] == (0.4 - 0.8) / 0.8


def test_relative_improvement_lower_is_better():
    """Log loss: lower variant score than baseline should count as a positive improvement."""
    baseline = {"task_a": 1.0}
    variant = {"task_a": 0.5}
    improvements = compute_relative_improvement(baseline, variant, higher_is_better=False)
    assert improvements["task_a"] == (1.0 - 0.5) / 1.0


def test_relative_improvement_normalizes_across_dataset_scales():
    """A fixed absolute gap should count for more on a dataset with a smaller baseline score."""
    baseline = {"small_scale": 0.1, "large_scale": 10.0}
    variant = {"small_scale": 0.2, "large_scale": 10.1}
    improvements = compute_relative_improvement(baseline, variant, higher_is_better=True)
    assert improvements["small_scale"] > improvements["large_scale"]


def test_relative_improvement_only_uses_common_tasks():
    baseline = {"task_a": 0.5, "task_only_in_baseline": 0.3}
    variant = {"task_a": 0.6, "task_only_in_variant": 0.9}
    improvements = compute_relative_improvement(baseline, variant, higher_is_better=True)
    assert set(improvements.keys()) == {"task_a"}


def test_significance_identical_scores_not_significant():
    baseline = {f"task_{i}": 0.7 for i in range(10)}
    variant = dict(baseline)
    result = paired_significance_test(baseline, variant)
    assert result["significant"] is False
    assert result["n_datasets"] == 10


def test_significance_detects_consistent_improvement():
    baseline = {f"task_{i}": 0.5 + 0.01 * i for i in range(10)}
    variant = {task_id: score + 0.1 for task_id, score in baseline.items()}
    result = paired_significance_test(baseline, variant)
    assert result["significant"] is True
    assert result["p_value"] < 0.05
    assert result["n_datasets"] == 10


def test_significance_noisy_tiny_improvement_not_significant():
    """A small, inconsistent difference across few datasets should not reach significance."""
    baseline = {"task_a": 0.70, "task_b": 0.55, "task_c": 0.62}
    variant = {"task_a": 0.71, "task_b": 0.54, "task_c": 0.63}
    result = paired_significance_test(baseline, variant)
    assert result["significant"] is False


def test_significance_requires_at_least_two_common_datasets():
    baseline = {"task_a": 0.5}
    variant = {"task_a": 0.6}
    result = paired_significance_test(baseline, variant)
    assert result["n_datasets"] == 1
    assert result["significant"] is False
    assert result["p_value"] != result["p_value"]  # NaN
