"""Tests for VRAM tracking (memory_stats.json) added for the activation ablation study."""

import json

import pytest
import torch

from nanotabpfn.experiment_utils import save_memory_stat
from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import train


def test_save_memory_stat_creates_file(tmp_path):
    """A fresh run_dir gets memory_stats.json with the given key/value."""
    save_memory_stat(tmp_path, "peak_vram_pretrain_gb", 4.2)

    stats_path = tmp_path / "memory_stats.json"
    assert stats_path.exists()
    with open(stats_path) as f:
        stats = json.load(f)
    assert stats == {"peak_vram_pretrain_gb": 4.2}


def test_save_memory_stat_merges_existing_keys(tmp_path):
    """Writing eval stats after pretrain stats must not clobber the earlier entry."""
    save_memory_stat(tmp_path, "peak_vram_pretrain_gb", 4.2)
    save_memory_stat(tmp_path, "peak_vram_eval_gb", 1.7)

    with open(tmp_path / "memory_stats.json") as f:
        stats = json.load(f)
    assert stats == {"peak_vram_pretrain_gb": 4.2, "peak_vram_eval_gb": 1.7}


def test_save_memory_stat_overwrites_same_key(tmp_path):
    """Re-saving the same key (e.g. a re-run) updates the value rather than duplicating it."""
    save_memory_stat(tmp_path, "peak_vram_pretrain_gb", 4.2)
    save_memory_stat(tmp_path, "peak_vram_pretrain_gb", 5.0)

    with open(tmp_path / "memory_stats.json") as f:
        stats = json.load(f)
    assert stats == {"peak_vram_pretrain_gb": 5.0}


def test_save_memory_stat_accepts_dict_value(tmp_path):
    """Per-dataset breakdowns (e.g. TabArena) are stored as nested dicts, not just scalars."""
    save_memory_stat(tmp_path, "peak_vram_eval_by_dataset_gb", {"adult": 1.1, "covertype": 2.3})

    with open(tmp_path / "memory_stats.json") as f:
        stats = json.load(f)
    assert stats["peak_vram_eval_by_dataset_gb"] == {"adult": 1.1, "covertype": 2.3}


def test_train_skips_memory_tracking_on_non_cuda_device(tmp_path):
    """On CPU/MPS (no CUDA), train() must not attempt CUDA memory calls or write memory_stats.json."""
    model = NanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=2,
        mlp_hidden_size=64,
        num_layers=1,
        num_outputs=2,
    )
    mock_prior = [
        {
            "x": torch.randn(2, 20, 5, dtype=torch.float32),
            "y": torch.randint(0, 2, (2, 20), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]
    checkpoint_dir = tmp_path / "checkpoints"

    train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=1,
        eval_func=None,
        checkpoint_dir=str(checkpoint_dir),
    )

    assert not (tmp_path / "memory_stats.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_train_writes_pretrain_vram_on_cuda(tmp_path):
    """On an actual CUDA device, train() must persist a positive peak_vram_pretrain_gb."""
    model = NanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=2,
        mlp_hidden_size=64,
        num_layers=1,
        num_outputs=2,
    )
    mock_prior = [
        {
            "x": torch.randn(2, 20, 5, dtype=torch.float32),
            "y": torch.randint(0, 2, (2, 20), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]
    checkpoint_dir = tmp_path / "checkpoints"

    train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cuda"),
        steps_per_eval=1,
        eval_func=None,
        checkpoint_dir=str(checkpoint_dir),
    )

    with open(tmp_path / "memory_stats.json") as f:
        stats = json.load(f)
    assert stats["peak_vram_pretrain_gb"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_run_experiment_writes_pretrain_and_eval_vram_on_cuda(tmp_path):
    """End-to-end: a real (tiny) run_experiment.py invocation persists both VRAM stats on GPU."""
    from run_experiment import parse_args, run_experiment

    args = parse_args(
        [
            "--seed",
            "0",
            "--num_steps",
            "2",
            "--eval_every",
            "2",
            "--checkpoint_every",
            "2",
            "--output_dir",
            str(tmp_path),
            "--activation",
            "gelu",
            "--batch_size",
            "2",
            "--max_seq_len",
            "50",
            "--max_features",
            "5",
            "--benchmark",
            "quick",
        ]
    )
    run_experiment(args)

    with open(tmp_path / "gelu" / "seed_0" / "memory_stats.json") as f:
        stats = json.load(f)
    assert stats["peak_vram_pretrain_gb"] > 0
    assert stats["peak_vram_eval_gb"] > 0
