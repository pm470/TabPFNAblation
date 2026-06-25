# Testing Standards

## Spec-Driven Testing

- Tests are derived from **acceptance criteria in spec files**, NEVER from implementation details
- Each acceptance criterion must map to at least one test assertion
- If a spec criterion lacks test coverage, that is a bug in the test suite

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
  - Correct parameter count for gated variants (SwiGLU, GeGLU, ReGLU must match non-gated param count)

### Experiment Output Tests
- Verify `config.json`, `metrics.jsonl`, and checkpoint files are created
- Verify JSONL metrics contain required fields: `step`, `wall_time`, `loss`

## Constraints

- All tests must pass on **CPU only** (no GPU required)
- Total test suite runtime must be **< 60 seconds** (use `--num_steps=10` or equivalent)
- Tests must not depend on `300k_150x5_2.h5` or any large data files
- Tests must not make network requests
