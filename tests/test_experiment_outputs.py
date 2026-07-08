"""Tests for experiment output structure and validity."""

import json
import shutil

# Use a temporary output dir for tests
import tempfile
from pathlib import Path

import pytest
import torch

from run_experiment import parse_args, run_experiment

TEST_OUTPUT_DIR = Path(tempfile.gettempdir()) / "tabpfn_results_test"


@pytest.fixture(autouse=True)
def cleanup_test_outputs():
    """Clean up test output directory before and after each test."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    yield
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    torch.cuda.empty_cache()


def run_short_experiment(seed=0, num_steps=10, eval_every=5, checkpoint_every=5):
    """Helper to run a short experiment for testing."""
    args = parse_args(
        [
            "--seed",
            str(seed),
            "--num_steps",
            str(num_steps),
            "--eval_every",
            str(eval_every),
            "--checkpoint_every",
            str(checkpoint_every),
            "--output_dir",
            str(TEST_OUTPUT_DIR),
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
    return run_experiment(args)


def test_config_json_created():
    """config.json exists and contains expected fields."""
    run_short_experiment()

    config_path = TEST_OUTPUT_DIR / "gelu" / "seed_0" / "config.json"
    assert config_path.exists(), f"config.json not found at {config_path}"

    with open(config_path) as f:
        config = json.load(f)

    expected_keys = {"activation", "seed", "num_steps", "batch_size", "lr", "param_count", "device"}
    assert expected_keys.issubset(config.keys()), f"Missing keys: {expected_keys - config.keys()}"
    assert config["activation"] == "gelu"
    assert config["seed"] == 0
    assert config["param_count"] > 0


def test_metrics_jsonl_valid():
    """metrics.jsonl exists, is valid JSONL, and has expected keys."""
    run_short_experiment(num_steps=10, eval_every=5)

    metrics_path = TEST_OUTPUT_DIR / "gelu" / "seed_0" / "metrics.jsonl"
    assert metrics_path.exists(), f"metrics.jsonl not found at {metrics_path}"

    entries = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                entries.append(entry)

    assert len(entries) > 0, "metrics.jsonl is empty"

    expected_keys = {"step", "wall_time", "loss", "roc_auc", "acc", "balanced_acc"}
    for entry in entries:
        assert expected_keys.issubset(entry.keys()), f"Missing keys in entry: {expected_keys - entry.keys()}"


def test_metrics_values_finite():
    """All metric values are finite (no NaN/Inf)."""
    run_short_experiment(num_steps=10, eval_every=5)

    metrics_path = TEST_OUTPUT_DIR / "gelu" / "seed_0" / "metrics.jsonl"
    with open(metrics_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            for key in ["loss", "roc_auc", "acc", "balanced_acc"]:
                val = entry[key]
                assert val == val, f"{key} is NaN"  # NaN != NaN
                assert abs(val) != float("inf"), f"{key} is Inf"


def test_checkpoints_created():
    """Checkpoint files are created at the expected intervals."""
    run_short_experiment(num_steps=10, eval_every=5, checkpoint_every=5)

    checkpoint_dir = TEST_OUTPUT_DIR / "gelu" / "seed_0" / "checkpoints"
    assert checkpoint_dir.exists(), f"Checkpoint dir not found at {checkpoint_dir}"

    # Should have step_00005.pt, step_00010.pt, and final.pt
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoint_files) >= 2, (
        f"Expected at least 2 checkpoints, got {len(checkpoint_files)}: {checkpoint_files}"
    )

    # final.pt must exist
    final_path = checkpoint_dir / "final.pt"
    assert final_path.exists(), "final.pt checkpoint not found"
