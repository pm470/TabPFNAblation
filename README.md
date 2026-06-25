# TabPFN Activation Function Ablation Study

Investigating whether modern activation functions (SwiGLU, GeGLU, Mish, etc.) can improve [nanoTabPFN](https://github.com/automl/nanoTabPFN) compared to the GELU baseline.

## Setup

Requires [mise](https://mise.jdx.dev/) and a ROCm-compatible AMD GPU.

```bash
mise install                # Python 3.11 + uv
uv sync --extra dev         # all dependencies
```

Download the prior data dump (~990MB) from [figshare](https://figshare.com/s/63fc1ada93e42e388e63) into the project root.

## Usage

```bash
# Single experiment (fast local breast_cancer eval)
python run_experiment.py --activation gelu --seed 0

# Single experiment with full TabArena evaluation at the end
python run_experiment.py --activation gelu --seed 0 --benchmark tabarena

# All seed × activation combos (default breast_cancer)
bash run_all.sh

# Plot results
python plot_results.py

# Tests & linting
uv run pytest -v
uv run ruff check .
```

Results are written to `results/<activation>/seed_<N>/` with `config.json`, `metrics.jsonl`, and `checkpoints/`.
