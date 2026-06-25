---
id: EXP-002
title: "Run Orchestration"
status: implemented
module: run_all.sh
last_synced: 2026-06-25
---

# Run Orchestration

## User Story

As a researcher, I want a single shell script that runs all activation × seed combinations sequentially so that I can launch a full ablation sweep with one command and have it fail fast on errors.

## Acceptance Criteria

### Shell Safety

- [ ] AC-1: Script starts with `#!/usr/bin/env bash` shebang.
- [ ] AC-2: Uses `set -euo pipefail` for strict error handling: exit on error (`-e`), undefined variable errors (`-u`), and pipeline failure propagation (`-o pipefail`).

### Working Directory

- [ ] AC-3: Resolves the script's own directory via `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`.
- [ ] AC-4: Changes to the script's directory with `cd "$SCRIPT_DIR"` before running experiments.

### Experiment Configuration

- [ ] AC-5: `ACTIVATIONS` array currently contains only `"gelu"`.
- [ ] AC-6: `SEEDS` array contains `0 1 2`.
- [ ] AC-7: `NUM_STEPS` is set to `2500`.
- [ ] AC-8: `EVAL_EVERY` is set to `25`.
- [ ] AC-9: `CHECKPOINT_EVERY` is set to `250`.

### Execution Loop

- [ ] AC-10: Iterates over all combinations of `ACTIVATIONS × SEEDS` (nested loops, activations outer, seeds inner).
- [ ] AC-11: Calls `python run_experiment.py` with `--activation`, `--seed`, `--num_steps`, `--eval_every`, and `--checkpoint_every` arguments.
- [ ] AC-12: Runs experiments sequentially (no parallelism).
- [ ] AC-13: Prints a header line `"--- Running: activation=${activation}, seed=${seed} ---"` before each run.

### Output

- [ ] AC-14: Prints a banner `"=== TabPFN Ablation Study ==="` with the configured activations, seeds, and step count at the start.
- [ ] AC-15: Prints `"=== All runs complete ==="` and directions to run `plot_results.py` at the end.

## Notes

- The script is designed for V1 validation: only GELU with 3 seeds. The `ACTIVATIONS` array will be expanded in later phases to include SwiGLU, GeGLU, Mish, etc.
- Sequential execution ensures reproducibility and simplifies debugging at the cost of wall-clock time.
- The `--benchmark` flag is not passed, so it defaults to `breast_cancer` (the fast local evaluation).
