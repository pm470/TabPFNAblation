"""Tests for experiment output structure and validity."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from run_experiment import parse_args, run_experiment


@pytest.fixture(scope="module")
def experiment_output_dir():
    """Run a single short experiment once for the entire module to check output files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="tabpfn_results_test_"))
    args = parse_args(
        [
            "--seed",
            "0",
            "--num_steps",
            "2",
            "--eval_every",
            "1",
            "--checkpoint_every",
            "1",
            "--output_dir",
            str(temp_dir),
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
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    torch.cuda.empty_cache()


def test_config_json_created(experiment_output_dir):
    """config.json exists and contains expected fields."""
    config_path = experiment_output_dir / "gelu" / "seed_0" / "config.json"
    assert config_path.exists(), f"config.json not found at {config_path}"

    with open(config_path) as f:
        config = json.load(f)

    expected_keys = {"activation", "seed", "num_steps", "batch_size", "lr", "param_count", "device"}
    assert expected_keys.issubset(config.keys()), f"Missing keys: {expected_keys - config.keys()}"
    assert config["activation"] == "gelu"
    assert config["seed"] == 0
    assert config["param_count"] > 0


def test_metrics_jsonl_valid(experiment_output_dir):
    """metrics.jsonl exists, is valid JSONL, and has expected keys."""
    metrics_path = experiment_output_dir / "gelu" / "seed_0" / "metrics.jsonl"
    assert metrics_path.exists(), f"metrics.jsonl not found at {metrics_path}"

    entries = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                entries.append(entry)

    assert len(entries) > 0, "metrics.jsonl is empty"

    expected_keys = {"step", "wall_time", "loss", "roc_auc", "acc", "balanced_acc", "param_count"}
    for entry in entries:
        assert expected_keys.issubset(entry.keys()), f"Missing keys in entry: {expected_keys - entry.keys()}"


def test_metrics_values_finite(experiment_output_dir):
    """All metric values are finite (no NaN/Inf)."""
    metrics_path = experiment_output_dir / "gelu" / "seed_0" / "metrics.jsonl"
    with open(metrics_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            for key in ["loss", "roc_auc", "acc", "balanced_acc"]:
                val = entry[key]
                assert val == val, f"{key} is NaN"  # NaN != NaN
                assert abs(val) != float("inf"), f"{key} is Inf"


def test_checkpoints_created(experiment_output_dir):
    """Checkpoint files are created at the expected intervals."""
    checkpoint_dir = experiment_output_dir / "gelu" / "seed_0" / "checkpoints"
    assert checkpoint_dir.exists(), f"Checkpoint dir not found at {checkpoint_dir}"

    # Should have step_00001.pt, step_00002.pt, and final.pt
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoint_files) >= 2, (
        f"Expected at least 2 checkpoints, got {len(checkpoint_files)}: {checkpoint_files}"
    )

    # final.pt must exist
    final_path = checkpoint_dir / "final.pt"
    assert final_path.exists(), "final.pt checkpoint not found"


def test_gated_unrestricted_directory_naming(tmp_path):
    """Gated activations with --gated-unrestricted save to {activation}_unrestricted directory."""
    args = parse_args(
        [
            "--seed",
            "0",
            "--num_steps",
            "1",
            "--eval_every",
            "1",
            "--checkpoint_every",
            "1",
            "--output_dir",
            str(tmp_path),
            "--activation",
            "swiglu",
            "--gated-unrestricted",
            "--batch_size",
            "2",
            "--max_seq_len",
            "50",
            "--max_features",
            "5",
        ]
    )
    run_experiment(args)
    unrestricted_dir = tmp_path / "swiglu_unrestricted" / "seed_0"
    assert unrestricted_dir.exists(), f"Expected directory {unrestricted_dir} to exist"
    assert (unrestricted_dir / "config.json").exists()
    with open(unrestricted_dir / "config.json") as f:
        config = json.load(f)
    assert config.get("gated_unrestricted") is True
