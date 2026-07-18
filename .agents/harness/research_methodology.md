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
