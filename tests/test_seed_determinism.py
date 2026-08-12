"""Test that seed control produces deterministic results."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from run_experiment import parse_args, run_experiment

TEST_OUTPUT_DIR = Path(tempfile.gettempdir()) / "tabpfn_results_test_determinism"


@pytest.fixture(autouse=True)
def cleanup_test_outputs():
    """Clean up test output directory before and after each test."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    yield
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    torch.cuda.empty_cache()


def run_and_collect_losses(seed, run_id):
    """Run an experiment and return the loss sequence."""
    output_dir = TEST_OUTPUT_DIR / f"run_{run_id}"
    args = parse_args(
        [
            "--seed",
            str(seed),
            "--num_steps",
            "2",
            "--eval_every",
            "1",
            "--checkpoint_every",
            "2",
            "--output_dir",
            str(output_dir),
            "--activation",
            "gelu",
            "--batch_size",
            "2",
            "--max_seq_len",
            "100",
            "--max_features",
            "10",
        ]
    )
    run_experiment(args)

    metrics_path = output_dir / "gelu" / f"seed_{seed}" / "metrics.jsonl"
    losses = []
    with open(metrics_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            losses.append(entry["loss"])
    torch.cuda.empty_cache()
    return losses


def test_same_seed_produces_identical_losses():
    """Two runs with the same seed must produce identical loss sequences."""
    losses_a = run_and_collect_losses(seed=42, run_id="a")
    losses_b = run_and_collect_losses(seed=42, run_id="b")

    assert len(losses_a) == len(losses_b), "Different number of eval steps"
    for i, (a, b) in enumerate(zip(losses_a, losses_b, strict=False)):
        assert a == b, f"Loss mismatch at eval step {i}: {a} != {b}"


def test_different_seeds_produce_different_losses():
    """Two runs with different seeds should produce different losses."""
    losses_a = run_and_collect_losses(seed=0, run_id="a")
    losses_b = run_and_collect_losses(seed=1, run_id="b")

    # At least some losses should differ
    assert any(a != b for a, b in zip(losses_a, losses_b, strict=False)), "Different seeds produced identical losses"
