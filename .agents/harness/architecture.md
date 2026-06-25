# Architecture Guidelines

## Module Responsibility Map

| Module | Responsibility |
|---|---|
| `model.py` | Neural network architecture: `NanoTabPFNModel`, encoders, transformer layers, decoder, `NanoTabPFNClassifier`. **Activation function changes go here ONLY.** |
| `prior.py` | Upstream nanoTabPFN prior data generation. **DO NOT MODIFY.** |
| `train.py` | Training loop, evaluation, `PriorDumpDataLoader`, seed control, device detection. |
| `run_experiment.py` | CLI experiment runner, argparse, structured logging, checkpoint orchestration. |
| `tabarena_eval.py` | TabArena benchmark wrapper (AutoGluon integration). |
| `plot_results.py` | Result visualization. Reads only from `results/` directory files. |
| `utils.py` | Shared utilities (currently empty, use for cross-module helpers). |

## Dependency Direction

```
run_experiment.py → train.py → model.py
tabarena_eval.py → model.py
plot_results.py → (reads files only, no code imports from other modules)
```

- **No circular imports.** If module A imports module B, module B must not import module A.
- `prior.py` is imported only by `train.py` (via `PriorDumpDataLoader`).

## Hyperparameter Flow

- All hyperparameters are defined via `argparse` in `run_experiment.py`
- `train.py` receives hyperparameters as function arguments — **never hardcode defaults in `train.py`**
- `model.py` receives architecture params via constructor arguments

## Adding New Activation Functions

1. Implement the activation class or function in `model.py`
2. Register it in the activation lookup/dispatch in `model.py`
3. No changes needed in `train.py` or `run_experiment.py` (activation name flows through argparse → model config)
4. Add smoke tests in `tests/`

## Results Directory Structure

```
results/
  {activation_name}/
    seed_{N}/
      config.json
      metrics.jsonl
      checkpoints/
        step_{S}.pt
```

- Never write results outside this structure
- Never overwrite existing results — rerun to regenerate
