# Research Methodology

## Fair Comparison Rules

- **Gated activations** (SwiGLU, GeGLU, ReGLU) must match the parameter count of non-gated variants
- Use the LLaMA-style trick: intermediate size `E → 2 * (2H/3)` so the gated projection has equivalent total parameters
- If parameter counts diverge, the comparison is invalid — verify with smoke tests before running experiments

## Frozen Hyperparameters

When comparing activation functions, **ONLY the activation changes**. Everything else stays at:

### Default Model Parameters (NanoTabPFN)
- **Embedding Size:** 128
- **Attention Heads:** 4
- **MLP Hidden:** 192
- **Layers:** 3
- **Outputs:** 10

### Default Training Parameters
- **Steps:** 5,000
- **Batch Size:** 8
- **Learning Rate:** 1e-3
- **Checkpointing:** Every 250 steps
- **Fast Evaluation:** Every 250 steps (3 proxy datasets)

Changing any of these invalidates the comparison unless explicitly studying that axis (e.g., depth ablation in phase 5).

## Seed Protocol

- **Development:** minimum 3 seeds per configuration
- **Final results:** 10 seeds per configuration
- Seeds must be explicitly set via `--seed` flag
- Full determinism: `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`

## Results & Logging

- Directory structure: `results/{activation}/seed_{N}/` containing `config.json`, `metrics.jsonl`, `checkpoints/`
- Metric logging: JSONL format with fields: `step`, `wall_time`, `loss`, plus eval metrics (`roc_auc`, `acc`, `balanced_acc`)
- **Never modify results after the fact** — rerun the experiment if corrections are needed
- Prior data: always use `300k_150x5_2.h5` for consistency across all experiments

## Statistical Reporting

- Report **mean ± std** across seeds for all metrics
- Use appropriate statistical tests (e.g., paired t-test, Wilcoxon) for final pairwise comparisons
- State the number of seeds and whether results are statistically significant (p < 0.05)

## Evaluation Strategy

| Stage | Benchmark | Purpose |
|---|---|---|
| Development | `diabetes`, `blood-transfusion`, `amazon_employee_access` | Fast iteration/sanity check (`--benchmark quick`) |
| Final results | TabArena | Publication-quality evaluation (`--benchmark tabarena`) |

- Do not draw conclusions from the development datasets — they are a sanity check only.
- TabArena runs use the `nanotabpfn` subset for testing, `classification` for final numbers.

## Phase 2: Architecture Ablation (Issue #26)

This phase studies how activations scale with architecture changes. Parameters differ from Phase 1.

### New Baseline (Phase 2)
- **Embedding Size:** 128
- **Attention Heads:** 4
- **MLP Hidden:** 256 (ratio = 2×, up from 1.5× in Phase 1)
- **Layers:** 3
- **Outputs:** 10

### Sweep Grid
| Axis | Values | Fixed params |
|------|--------|-------------|
| **Layers** | [1, 2, 3, 4, 6] | E=128, H=256 |
| **Hidden** | [64, 128, 256, 384, 512] | E=128, L=3 |
| **Embedding** | [64, 128, 192, 256] | H=256, L=3 |

### Training Configuration (Phase 2)
- **Steps:** 10,000 forward passes (= 5,000 optimizer steps with accumulation)
- **Batch Size:** 4 (micro), accumulation_steps=2 (effective batch = 8)
- **Gradient Checkpointing:** Always enabled
- **Learning Rate:** 1e-3 (same as Phase 1)
- **Evaluation:** Quick eval only (3 proxy datasets, every 250 steps)
- **Checkpointing:** Every 250 steps

### Activations Evaluated
- GELU (baseline), SwiGLU, Bilinear (top 2 from Phase 1)

### Seeds
- 3 seeds per configuration (seeds 0, 1, 2)
- Total: 14 architecture points × 3 activations × 3 seeds = 126 jobs

### Results Directory
- `results_arch/{sweep_axis}/{activation}/e{E}_h{H}_l{L}/seed_{N}/`
