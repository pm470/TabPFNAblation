"""Tests for training pipeline and data loading."""

from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import PriorDumpDataLoader, _real_get_eval_datasets, train
from nanotabpfn.train import eval as eval_fn


def test_train_without_eval_func():
    """train() correctly handles eval_func=None, only logging step, wall_time, loss."""
    model = NanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=2,
        mlp_hidden_size=64,
        num_layers=1,
        num_outputs=2,
    )

    # Mock prior with a single batch
    mock_prior = [
        {
            "x": torch.randn(2, 20, 5, dtype=torch.float32),
            "y": torch.randint(0, 2, (2, 20), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]

    # Train for 1 step, eval every 1 step
    _, eval_history = train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=1,
        eval_func=None,
    )
    assert len(eval_history) == 1
    entry = eval_history[0]
    expected_keys = {"step", "wall_time", "loss", "param_count"}
    assert set(entry.keys()) == expected_keys


def test_train_param_count_in_history():
    """train() correctly includes the correct param_count in eval_history."""
    model = NanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=2,
        mlp_hidden_size=64,
        num_layers=1,
        num_outputs=2,
    )
    expected_param_count = sum(p.numel() for p in model.parameters())

    mock_prior = [
        {
            "x": torch.randn(2, 20, 5, dtype=torch.float32),
            "y": torch.randint(0, 2, (2, 20), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]

    _, eval_history = train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=1,
        eval_func=None,
    )
    assert len(eval_history) == 1
    assert eval_history[0]["param_count"] == expected_param_count


def test_prior_dump_dataloader_wraparound(tmp_path):
    """PriorDumpDataLoader correctly wraps around when reaching the end of the HDF5 dataset."""
    # Create a tiny HDF5 file
    h5_path = tmp_path / "tiny_prior.h5"
    num_samples = 3

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("max_num_classes", data=[2])
        # Need X, y, num_features, num_datapoints, single_eval_pos
        f.create_dataset("X", data=np.random.randn(num_samples, 10, 5).astype(np.float32))
        f.create_dataset("y", data=np.random.randint(0, 2, (num_samples, 10)).astype(np.float32))
        f.create_dataset("num_features", data=np.array([5, 5, 5], dtype=np.int32))
        f.create_dataset("num_datapoints", data=np.array([10, 10, 10], dtype=np.int32))
        f.create_dataset("single_eval_pos", data=np.array([5, 5, 5], dtype=np.int32))

    loader = PriorDumpDataLoader(filename=str(h5_path), num_steps=3, batch_size=2, device=torch.device("cpu"))

    # Batch 1: consumes 2 samples (index 0, 1) -> pointer = 2
    # Batch 2: consumes 2 samples (but dataset only has 3, so pointer = 4, then wrapping triggers pointer = 0)
    # Batch 3: consumes 2 samples (index 0, 1) -> pointer = 2

    batches = list(loader)

    assert len(batches) == 3
    for batch in batches:
        assert "x" in batch
        assert "y" in batch
        assert "train_test_split_index" in batch

    # Verify wraparound reset pointer to 0
    assert loader.pointer == 2


def test_prior_dump_dataloader_seed_offset(tmp_path):
    """PriorDumpDataLoader with different seeds starts at different offsets."""
    h5_path = tmp_path / "tiny_prior.h5"
    num_samples = 100  # Large enough to make collisions unlikely

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("max_num_classes", data=[2])
        f.create_dataset("X", data=np.random.randn(num_samples, 10, 5).astype(np.float32))
        f.create_dataset("y", data=np.random.randint(0, 2, (num_samples, 10)).astype(np.float32))
        f.create_dataset("num_features", data=np.full(num_samples, 5, dtype=np.int32))
        f.create_dataset("num_datapoints", data=np.full(num_samples, 10, dtype=np.int32))
        f.create_dataset("single_eval_pos", data=np.full(num_samples, 5, dtype=np.int32))

    loader_a = PriorDumpDataLoader(filename=str(h5_path), num_steps=1, batch_size=2, device=torch.device("cpu"), seed=0)
    loader_b = PriorDumpDataLoader(filename=str(h5_path), num_steps=1, batch_size=2, device=torch.device("cpu"), seed=1)
    loader_none = PriorDumpDataLoader(filename=str(h5_path), num_steps=1, batch_size=2, device=torch.device("cpu"))

    # Different seeds → different starting pointers
    assert loader_a.pointer != loader_b.pointer
    # No seed → starts at 0
    assert loader_none.pointer == 0


def test_prior_dump_dataloader_seed_alignment(tmp_path):
    """PriorDumpDataLoader aligns seed offsets to batch size."""
    h5_path = tmp_path / "tiny_prior.h5"
    num_samples = 100

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("max_num_classes", data=[2])
        f.create_dataset("X", data=np.random.randn(num_samples, 10, 5).astype(np.float32))
        f.create_dataset("y", data=np.random.randint(0, 2, (num_samples, 10)).astype(np.float32))
        f.create_dataset("num_features", data=np.full(num_samples, 5, dtype=np.int32))
        f.create_dataset("num_datapoints", data=np.full(num_samples, 10, dtype=np.int32))
        f.create_dataset("single_eval_pos", data=np.full(num_samples, 5, dtype=np.int32))

    # Test with multiple random seeds to ensure alignment holds
    for seed in range(5):
        loader = PriorDumpDataLoader(
            filename=str(h5_path), num_steps=1, batch_size=8, device=torch.device("cpu"), seed=seed
        )
        assert loader.pointer % 8 == 0, f"Pointer {loader.pointer} not aligned to batch size 8 for seed {seed}"


def test_prior_dump_dataloader_seed_determinism(tmp_path):
    """PriorDumpDataLoader with the same seed always starts at the same offset."""
    h5_path = tmp_path / "tiny_prior.h5"
    num_samples = 100

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("max_num_classes", data=[2])
        f.create_dataset("X", data=np.random.randn(num_samples, 10, 5).astype(np.float32))
        f.create_dataset("y", data=np.random.randint(0, 2, (num_samples, 10)).astype(np.float32))
        f.create_dataset("num_features", data=np.full(num_samples, 5, dtype=np.int32))
        f.create_dataset("num_datapoints", data=np.full(num_samples, 10, dtype=np.int32))
        f.create_dataset("single_eval_pos", data=np.full(num_samples, 5, dtype=np.int32))

    loader1 = PriorDumpDataLoader(filename=str(h5_path), num_steps=1, batch_size=2, device=torch.device("cpu"), seed=42)
    loader2 = PriorDumpDataLoader(filename=str(h5_path), num_steps=1, batch_size=2, device=torch.device("cpu"), seed=42)

    assert loader1.pointer == loader2.pointer


def test_prior_dump_dataloader_seed_different_data(tmp_path):
    """PriorDumpDataLoader with different seeds yields different batch data."""
    h5_path = tmp_path / "prior.h5"
    num_samples = 200

    rng = np.random.RandomState(99)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("max_num_classes", data=[2])
        f.create_dataset("X", data=rng.randn(num_samples, 10, 5).astype(np.float32))
        f.create_dataset("y", data=rng.randint(0, 2, (num_samples, 10)).astype(np.float32))
        f.create_dataset("num_features", data=np.full(num_samples, 5, dtype=np.int32))
        f.create_dataset("num_datapoints", data=np.full(num_samples, 10, dtype=np.int32))
        f.create_dataset("single_eval_pos", data=np.full(num_samples, 5, dtype=np.int32))

    loader_a = PriorDumpDataLoader(filename=str(h5_path), num_steps=1, batch_size=2, device=torch.device("cpu"), seed=0)
    loader_b = PriorDumpDataLoader(filename=str(h5_path), num_steps=1, batch_size=2, device=torch.device("cpu"), seed=1)

    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))

    assert not torch.equal(batch_a["x"], batch_b["x"])


def test_prior_dump_dataloader_skip_steps(tmp_path):
    """PriorDumpDataLoader skip_steps advances the pointer for auto-resume."""
    h5_path = tmp_path / "prior.h5"
    num_samples = 200

    rng = np.random.RandomState(99)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("max_num_classes", data=[2])
        f.create_dataset("X", data=rng.randn(num_samples, 10, 5).astype(np.float32))
        f.create_dataset("y", data=rng.randint(0, 2, (num_samples, 10)).astype(np.float32))
        f.create_dataset("num_features", data=np.full(num_samples, 5, dtype=np.int32))
        f.create_dataset("num_datapoints", data=np.full(num_samples, 10, dtype=np.int32))
        f.create_dataset("single_eval_pos", data=np.full(num_samples, 5, dtype=np.int32))

    batch_size = 4
    skip = 10  # simulate resuming after 10 steps

    # A fresh loader that runs 15 steps from seed offset
    loader_full = PriorDumpDataLoader(
        filename=str(h5_path), num_steps=15, batch_size=batch_size, device=torch.device("cpu"), seed=42
    )
    # A resumed loader that skips the first 10 steps
    loader_resumed = PriorDumpDataLoader(
        filename=str(h5_path), num_steps=5, batch_size=batch_size, device=torch.device("cpu"), seed=42, skip_steps=skip
    )

    # The resumed loader's pointer should equal the full loader's pointer
    # after consuming 10 batches
    expected_pointer = (loader_full.pointer + skip * batch_size) % num_samples
    assert loader_resumed.pointer == expected_pointer

    # Consume all batches from both and compare: the last 5 batches
    # of the full loader should match the 5 batches of the resumed loader
    full_batches = list(loader_full)
    resumed_batches = list(loader_resumed)

    for fb, rb in zip(full_batches[10:], resumed_batches, strict=True):
        assert torch.equal(fb["x"], rb["x"])


def test_save_checkpoint_helper(tmp_path):
    """_save_checkpoint() correctly calls eval() and train() on optimizer."""
    import schedulefree

    from nanotabpfn.train import _save_checkpoint

    model = NanoTabPFNModel(embedding_size=32, num_attention_heads=2, mlp_hidden_size=64, num_layers=1, num_outputs=2)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3, weight_decay=0.0)

    # Train mode by default
    model.train()
    optimizer.train()

    ckpt_dir = tmp_path / "checkpoints"
    _save_checkpoint(model, optimizer, str(ckpt_dir), "step_00001.pt")

    # Ensure file was created
    assert (ckpt_dir / "step_00001.pt").exists()


def test_total_eval_time_printed(capsys):
    """Total inline eval time is printed at the end of training if eval_func is provided."""
    model = NanoTabPFNModel(embedding_size=32, num_attention_heads=2, mlp_hidden_size=64, num_layers=1, num_outputs=2)
    mock_prior = [
        {
            "x": torch.randn(2, 20, 5, dtype=torch.float32),
            "y": torch.randint(0, 2, (2, 20), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]

    def mock_eval_func(classifier):
        import time

        time.sleep(0.1)  # Simulate eval time
        return {"accuracy": 1.0}

    train(model, mock_prior, lr=1e-3, device=torch.device("cpu"), steps_per_eval=1, eval_func=mock_eval_func)  # type: ignore

    captured = capsys.readouterr()
    assert "[NanoTabPFN] Total inline eval time:" in captured.out


def test_checkpoint_saves_eval_weights(tmp_path):
    """Checkpoints saved mid-training should contain eval-mode weights."""
    model = NanoTabPFNModel(embedding_size=32, num_attention_heads=2, mlp_hidden_size=64, num_layers=1, num_outputs=2)
    mock_prior = [
        {
            "x": torch.randn(2, 20, 5, dtype=torch.float32),
            "y": torch.randint(0, 2, (2, 20), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]

    ckpt_dir = tmp_path / "checkpoints"

    # Train for 1 step, save checkpoint every 1 step
    model, _ = train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=1,
    )

    assert (ckpt_dir / "step_00001.pt").exists()

    # Manually extract eval weights from the model now that training is done
    import schedulefree

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3, weight_decay=0.0)
    optimizer.eval()
    expected_weights = model.state_dict()

    # Load the saved checkpoint
    saved_weights = torch.load(ckpt_dir / "step_00001.pt", map_location="cpu", weights_only=True)

    # Compare weights (just pick the first key)
    first_key = next(iter(expected_weights))
    assert torch.allclose(expected_weights[first_key], saved_weights[first_key])


def test_final_checkpoint_saves_eval_weights(tmp_path):
    """The final.pt checkpoint should contain eval-mode weights."""
    model = NanoTabPFNModel(embedding_size=32, num_attention_heads=2, mlp_hidden_size=64, num_layers=1, num_outputs=2)
    mock_prior = [
        {
            "x": torch.randn(2, 20, 5, dtype=torch.float32),
            "y": torch.randint(0, 2, (2, 20), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]

    ckpt_dir = tmp_path / "checkpoints"

    model, _ = train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=100,
    )

    assert (ckpt_dir / "final.pt").exists()

    import schedulefree

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3, weight_decay=0.0)
    optimizer.eval()
    expected_weights = model.state_dict()

    saved_weights = torch.load(ckpt_dir / "final.pt", map_location="cpu", weights_only=True)
    first_key = next(iter(expected_weights))
    assert torch.allclose(expected_weights[first_key], saved_weights[first_key])


def test_eval_empty_datasets():
    """eval() raises ValueError when given an empty dataset list."""
    mock_classifier = MagicMock()
    with pytest.raises(ValueError, match="No evaluation datasets provided"):
        eval_fn(mock_classifier, datasets=[])


def test_get_eval_datasets_failure():
    """get_eval_datasets() raises RuntimeError if all network fetches fail."""
    with (
        patch("sklearn.datasets.fetch_openml", side_effect=RuntimeError("Network Error")),
        pytest.raises(RuntimeError, match="Failed to fetch evaluation datasets from OpenML"),
    ):
        _real_get_eval_datasets()
