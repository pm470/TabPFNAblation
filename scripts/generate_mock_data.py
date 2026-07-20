"""Generate synthetic experiment results for a TabPFN activation function ablation study.

Produces mock data into ``results_mock/`` that mirrors the directory layout
expected by :func:`scripts.plot_results.load_all_results`:

    results_mock/{activation}/seed_{N}/metrics.jsonl
    results_mock/{activation}/seed_{N}/config.json
    results_mock/architecture_ablation.json

Usage::

    python scripts/generate_mock_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.random import Generator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVATION_PROFILES: dict[str, dict[str, float]] = {
    "gelu": {"final_auc": 0.72, "convergence_rate": 800, "noise_std": 0.015},
    "relu": {"final_auc": 0.69, "convergence_rate": 900, "noise_std": 0.020},
    "swish": {"final_auc": 0.73, "convergence_rate": 750, "noise_std": 0.014},
    "leaky_relu": {"final_auc": 0.70, "convergence_rate": 880, "noise_std": 0.018},
    "prelu": {"final_auc": 0.71, "convergence_rate": 860, "noise_std": 0.017},
    "swiglu": {"final_auc": 0.76, "convergence_rate": 650, "noise_std": 0.012},
    "bilinear": {"final_auc": 0.74, "convergence_rate": 700, "noise_std": 0.013},
}

NUM_SEEDS: int = 10
STEP_START: int = 25
STEP_END: int = 5000
STEP_INCREMENT: int = 25

CONFIG_TEMPLATE: dict[str, object] = {
    "num_steps": 5000,
    "batch_size": 32,
    "lr": 0.004,
    "eval_every": 25,
    "checkpoint_every": 250,
    "embedding_size": 128,
    "num_attention_heads": 4,
    "mlp_hidden_size": 192,
    "num_layers": 3,
    "num_outputs": 10,
    "param_count": 574218,
    "device": "cuda",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_steps() -> np.ndarray:
    """Return the array of evaluation steps from 25 to 5000 inclusive."""
    return np.arange(STEP_START, STEP_END + 1, STEP_INCREMENT)


def generate_learning_curve(
    steps: np.ndarray,
    final_auc: float,
    convergence_rate: float,
    noise_std: float,
    rng: Generator,
) -> np.ndarray:
    """Generate a single ROC-AUC learning curve for one seed.

    Args:
        steps: 1-D array of training steps at which to evaluate.
        final_auc: Asymptotic AUC the curve approaches.
        convergence_rate: Characteristic step count controlling curve shape.
        noise_std: Base standard-deviation used to derive per-seed and
            per-step noise magnitudes.
        rng: NumPy random generator instance.

    Returns:
        1-D array of AUC values, one per step, clipped to [0.5, 1.0].
    """
    per_seed_offset: float = float(rng.normal(0, noise_std / 2))
    per_step_noise: np.ndarray = rng.normal(0, noise_std / 3, size=len(steps))
    auc = final_auc * (1 - np.exp(-steps / convergence_rate)) + per_seed_offset + per_step_noise
    return np.clip(auc, 0.5, 1.0)


def generate_loss_curve(steps: np.ndarray, rng: Generator) -> np.ndarray:
    """Generate a monotonically-decreasing loss curve.

    Args:
        steps: 1-D array of training steps.
        rng: NumPy random generator instance.

    Returns:
        1-D array of loss values, one per step.
    """
    noise = rng.normal(0, 0.02, size=len(steps))
    loss = 2.5 * np.exp(-steps / 600) + 0.5 + noise
    return loss


def generate_wall_times(steps: np.ndarray, rng: Generator) -> np.ndarray:
    """Generate monotonically-increasing wall-clock times.

    Args:
        steps: 1-D array of training steps.
        rng: NumPy random generator instance.

    Returns:
        1-D cumulative wall-time array (seconds).
    """
    base_dt = 0.35  # seconds per eval interval on average
    increments = base_dt + rng.exponential(0.05, size=len(steps))
    return np.cumsum(increments)


# ---------------------------------------------------------------------------
# Per-seed data generation
# ---------------------------------------------------------------------------


def write_seed_data(
    activation: str,
    seed: int,
    output_dir: Path,
    rng: Generator,
) -> None:
    """Write ``metrics.jsonl`` and ``config.json`` for a single seed run.

    Args:
        activation: Name of the activation function.
        seed: Seed index (0-based).
        output_dir: Root output directory (e.g. ``results_mock/``).
        rng: NumPy random generator instance.
    """
    seed_dir = output_dir / activation / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    profile = ACTIVATION_PROFILES[activation]
    steps = _make_steps()

    auc_values = generate_learning_curve(
        steps,
        final_auc=profile["final_auc"],
        convergence_rate=profile["convergence_rate"],
        noise_std=profile["noise_std"],
        rng=rng,
    )
    loss_values = generate_loss_curve(steps, rng)
    wall_times = generate_wall_times(steps, rng)

    # --- metrics.jsonl ---
    metrics_path = seed_dir / "metrics.jsonl"
    with metrics_path.open("w") as f:
        for i, step in enumerate(steps):
            record = {
                "step": int(step),
                "wall_time": round(float(wall_times[i]), 4),
                "loss": round(float(loss_values[i]), 6),
                "param_count": CONFIG_TEMPLATE["param_count"],
                "roc_auc": round(float(auc_values[i]), 6),
            }
            f.write(json.dumps(record) + "\n")

    # --- config.json ---
    config = {
        "activation": activation,
        "seed": seed,
        **CONFIG_TEMPLATE,
    }
    config_path = seed_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Architecture ablation data
# ---------------------------------------------------------------------------


def generate_architecture_ablation(output_dir: Path, rng: Generator) -> None:
    """Generate ``architecture_ablation.json`` with depth and width scaling data.

    Args:
        output_dir: Root output directory (e.g. ``results_mock/``).
        rng: NumPy random generator instance.
    """
    activations = ["gelu", "swiglu", "bilinear"]

    # --- Depth scaling ---
    layers = [1, 2, 3, 4, 6, 8]
    # Base asymptotes per activation (ordered as *activations*).
    depth_bases = {"gelu": 0.60, "swiglu": 0.62, "bilinear": 0.61}
    # Per-layer gain (diminishing).  SwiGLU/bilinear benefit more from depth.
    depth_gain = {"gelu": 0.025, "swiglu": 0.032, "bilinear": 0.030}

    depth_results: dict[str, dict[str, list[float]]] = {}
    for act in activations:
        means: list[float] = []
        stds: list[float] = []
        for n_layers in layers:
            mean_val = depth_bases[act] + depth_gain[act] * np.log1p(n_layers) + float(rng.normal(0, 0.005))
            mean_val = float(np.clip(mean_val, 0.55, 0.82))
            # Std shrinks for deeper models.
            std_val = 0.025 - 0.002 * n_layers + float(rng.uniform(0, 0.003))
            std_val = float(np.clip(std_val, 0.008, 0.025))
            means.append(round(mean_val, 4))
            stds.append(round(std_val, 4))
        depth_results[act] = {"mean": means, "std": stds}

    # --- Width scaling ---
    hidden_dims = [64, 128, 192, 256, 384, 512]
    width_bases = {"gelu": 0.58, "swiglu": 0.60, "bilinear": 0.59}
    width_gain = {"gelu": 0.028, "swiglu": 0.032, "bilinear": 0.030}

    width_results: dict[str, dict[str, list[float]]] = {}
    for act in activations:
        means = []
        stds = []
        for dim in hidden_dims:
            mean_val = width_bases[act] + width_gain[act] * np.log2(dim / 64) + float(rng.normal(0, 0.005))
            mean_val = float(np.clip(mean_val, 0.58, 0.80))
            std_val = 0.022 - 0.002 * np.log2(dim / 64) + float(rng.uniform(0, 0.003))
            std_val = float(np.clip(std_val, 0.008, 0.025))
            means.append(round(mean_val, 4))
            stds.append(round(std_val, 4))
        width_results[act] = {"mean": means, "std": stds}

    ablation_data = {
        "depth": {
            "layers": layers,
            "results": depth_results,
        },
        "width": {
            "hidden_dims": hidden_dims,
            "results": width_results,
        },
    }

    out_path = output_dir / "architecture_ablation.json"
    with out_path.open("w") as f:
        json.dump(ablation_data, f, indent=2)
        f.write("\n")

    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: Path | None = None) -> None:
    """Generate all mock experiment data.

    Args:
        output_dir: Root directory to write results into.  Defaults to
            ``results_mock/`` relative to the project root (two levels up
            from this script).
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "results_mock"

    output_dir.mkdir(parents=True, exist_ok=True)
    rng: Generator = np.random.default_rng(42)

    print(f"Generating mock data in {output_dir}\n")

    for activation, profile in ACTIVATION_PROFILES.items():
        print(f"Activation: {activation}  (final_auc={profile['final_auc']}, τ={profile['convergence_rate']})")
        for seed in range(NUM_SEEDS):
            write_seed_data(activation, seed, output_dir, rng)
            print(f"  seed_{seed} ✓")
        print()

    print("Generating architecture ablation data …")
    generate_architecture_ablation(output_dir, rng)

    total_seeds = len(ACTIVATION_PROFILES) * NUM_SEEDS
    print(f"\nDone — {total_seeds} runs across {len(ACTIVATION_PROFILES)} activations.")


if __name__ == "__main__":
    main()
