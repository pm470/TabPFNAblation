"""Utility functions."""

import json
from pathlib import Path


def save_memory_stat(run_dir: Path, key: str, value: float | dict) -> None:
    """Merge a single peak-VRAM measurement into run_dir/memory_stats.json.

    Read-modify-write so pretrain and eval stats (written at different points
    in the pipeline) accumulate into the same file instead of overwriting each other.
    """
    stats_path = Path(run_dir) / "memory_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    stats[key] = value
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_env(env_path: str | Path = ".env") -> None:
    """Load environment variables from a .env file."""
    import os

    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = value
