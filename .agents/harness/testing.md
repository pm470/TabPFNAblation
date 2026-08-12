# Testing Standards

## Spec-Driven Testing

The project divides its components into two categories for testing: **Core** (model architecture, training loop, seed determinism) and **Non-Core** (plotting, CLI, evaluation wrappers).

### Core Components
- Tests are derived from **acceptance criteria in spec files**, NEVER from implementation details.
- Each acceptance criterion for a core component must explicitly map to at least one test assertion.
- If a core spec criterion lacks test coverage, that is a bug in the test suite. We do not enforce a strict coverage percentage, but all core ACs must be covered.

### Non-Core Components
- We do NOT require strict test mapping for non-core components. 100% test coverage is unrealistic and overkill for these scripts.
- Acceptance criteria for non-core components can be checked off via manual verification or implicitly through end-to-end execution.

## Test Organization

- All tests live in `tests/` with `test_` prefix
- File naming: `test_{module_name}.py` or `test_{feature}.py`
- Use pytest fixtures for setup, teardown, and shared state
- Use `tmp_path` fixture or dedicated test output dirs with cleanup

## Existing Test Patterns

Follow these as reference implementations:

| File | Purpose |
|---|---|
| `tests/test_model_smoke.py` | Activation smoke tests (shape, finite values, param counts) |
| `tests/test_seed_determinism.py` | Reproducibility verification across identical seeds |
| `tests/test_experiment_outputs.py` | End-to-end output structure validation |

## Required Test Categories

### Seed Determinism
- **Mandatory** for any model or training change
- Two runs with identical seed must produce identical outputs
- Verified with `torch.allclose` on model outputs

### Activation Smoke Tests
- Required for every new activation function
- Must verify:
  - Output shape matches input shape (or expected gated output shape)
  - All output values are finite (no NaN/Inf)
  - Correct parameter count for gated variants (SwiGLU, ReGLU must match non-gated param count)

### Experiment Output Tests
- Verify `config.json`, `metrics.jsonl`, and checkpoint files are created
- Verify JSONL metrics contain required fields: `step`, `wall_time`, `loss`

## Constraints

- All tests must pass on **CPU only** (no GPU required)
- Total test suite runtime must be **< 60 seconds** (use `--num_steps=10` or equivalent)
- Tests must not depend on `300k_150x5_2.h5` or any large data files
- Tests must not make network requests
