"""Tests for Slurm cluster submission scripts."""

import subprocess


def test_submit_chunk_dry_run_4():
    """Verify that submit_chunk.sh dry-run for chunk 4 runs cleanly and sets GATED_UNRESTRICTED=1."""
    cmd = ["bash", "slurm/submit_chunk.sh", "--dry-run", "4"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    stdout = result.stdout
    assert "Submitting Chunk 4" in stdout
    assert "swiglu" in stdout
    assert "bilinear" in stdout
    assert 'GATED_UNRESTRICTED="1"' in stdout
