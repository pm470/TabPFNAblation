---
id: EXP-002
title: "Run Orchestration"
status: implemented
module: run_all.sh
last_synced: 2026-07-15
---

# Run Orchestration

## User Story

As a researcher, I want a single shell script that runs all activation × seed combinations sequentially so that I can launch a full ablation sweep with one command and have it fail fast on errors.

## Acceptance Criteria

### Shell Safety

- [x] AC-1: Script starts with `#!/usr/bin/env bash` shebang.
- [x] AC-2: Uses `set -euo pipefail` for strict error handling: exit on error (`-e`), undefined variable errors (`-u`), and pipeline failure propagation (`-o pipefail`).

### Working Directory

- [x] AC-3: Resolves the script's own directory via `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`.
- [x] AC-4: Changes to the script's directory with `cd "$SCRIPT_DIR"` before running experiments.

### Experiment Configuration

- [x] AC-5: `ACTIVATIONS` array contains `"bilinear" "swiglu" "relu" "gelu" "swish" "prelu" "leaky_relu"`.
- [x] AC-6: `SEEDS` array contains `0 1 2`.
- [x] AC-7: `NUM_STEPS` is set to `2500`.
- [x] AC-8: `EVAL_EVERY` is set to `25`.
- [x] AC-9: `CHECKPOINT_EVERY` is set to `250`.

### Execution Loop

- [x] AC-10: Iterates over all combinations of `ACTIVATIONS × SEEDS` (nested loops, activations outer, seeds inner).
- [x] AC-11: Calls `python run_experiment.py` with `--activation`, `--seed`, `--num_steps`, `--eval_every`, and `--checkpoint_every` arguments.
- [x] AC-12: Runs experiments sequentially (no parallelism).
- [x] AC-13: Prints a header line `"--- Running: activation=${activation}, seed=${seed} ---"` before each run.

### Output

- [x] AC-14: Prints a banner `"=== TabPFN Ablation Study ==="` with the configured activations, seeds, and step count at the start.
- [x] AC-15: Prints `"=== All runs complete ==="` and directions to run `plot_results.py` at the end.

## Notes

- The script is configured to test the core non-gated activations (GELU, ReLU, Swish, PReLU, Leaky ReLU) and gated variants (SwiGLU, Bilinear).
- Sequential execution ensures reproducibility and simplifies debugging at the cost of wall-clock time.
- The `--benchmark` flag is not passed, so it defaults to `breast_cancer` (the fast local evaluation).
