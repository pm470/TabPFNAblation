# TabPFN Activation Function Ablation Study

## Development Workflow (MANDATORY)

Every code change must follow this process. Do not skip steps.

### 1. Understand → 2. Consult Guidelines → 3. Check Spec → 4. Implement → 5. Verify → 6. Sync Spec

1. **Understand** the task fully before writing any code.
2. **Read the relevant guideline** (lazy-load from `.agents/harness/`):
   - Coding changes → read `.agents/harness/coding.md`
   - Architecture or module changes → read `.agents/harness/architecture.md`
   - Writing or modifying tests → read `.agents/harness/testing.md`
   - Experiment design or running experiments → read `.agents/harness/research_methodology.md`
3. **Check/Draft the relevant spec** in `specs/` — understand the acceptance criteria for the code being changed. **CRITICAL:** If no spec exists, or if an existing spec needs to be updated to cover new requirements, you MUST write or update the spec *first* (with `status: new` or `status: edited`) before writing any code.
4. **Implement** the change. If writing new functionality, write tests first (derived from acceptance criteria, not from implementation).
5. **Run verification**: `bash scripts/verify.sh` — fix ALL failures before considering the task done.
6. **Sync the spec**: update the corresponding spec file to reflect the current behavior. Set `status: implemented` and `last_synced` to today's date.

### Spec Sync Rules

- After ANY code change, the corresponding spec file in `specs/` must be updated to match.
- After ANY spec change without a code change, set `status: edited` in the spec frontmatter.
- Checkboxes (`[x]`) must be used to track met criteria. Acceptance criteria can **only** be checked off if they have been explicitly verified against the codebase (e.g., via automated tests or manual inspection). When `status: implemented`, all AC checkboxes must be checked.
- Never report a task as "done" if specs and code are out of sync.
- Tests must verify acceptance criteria from specs, not implementation details.

### Harness Correction

If the agent produces output that technically passes verification but doesn't match project intent, the guideline files in `.agents/harness/` should be corrected — not just the code.

---

## Agent Workspace Rules

- **Scratch Files**: ANY temporary Python scripts, test scripts, or one-off data files created by the agent for intermediate calculations or testing MUST be stored in the agent's designated persistent scratch directory (`<appDataDir>/brain/<conversation-id>/scratch/`). They MUST NOT be placed in the project root or any codebase directory to avoid cluttering the workspace or syncing to the cluster.

---

## Project Context

This is a CS Master's research project (DLL course) investigating whether modern activation functions (e.g., SwiGLU, GeGLU, Mish) can improve tabular foundation models like TabPFN compared to the GELU baseline.

## Hypothesis

Modern activation functions might yield statistically significant improvements compared to the GELU baseline on the TabArena benchmark.

## Codebase

- Based on **nanoTabPFN** (<https://github.com/automl/nanoTabPFN>) — a simplified ~500 LOC reimplementation of TabPFNv2.
- `model.py` and `prior.py` are from upstream nanoTabPFN. `prior.py` is excluded from ruff linting.
- `train.py` has been enhanced with full seed control, checkpoint saving, and structured JSONL metric logging.
- `run_experiment.py` is the CLI experiment runner. Uses `--benchmark` flag to switch between fast local and TabArena eval.
- `tabarena_eval.py` wraps the model for the official TabArena Autogluon benchmark pipeline.
- `run_all.sh` runs all seed × activation combos.
- Prior data dump: `300k_150x5_2.h5` (990MB, downloaded from figshare, not committed to git).

## Hardware

- **Local:** Ryzen 5950X + Radeon RX 9700 XT (16GB VRAM), ROCm 7.2, PyTorch 2.11.0+rocm7.2
- **Cluster:** BwUniClust3.0 (for full TabArena evaluation — needs 48GB VRAM recommended)

## Environment

- Python 3.11.15 via mise, dependencies managed with uv
- PyTorch ROCm 7.2 wheel from `https://download.pytorch.org/whl/rocm7.2`
- Dev tools: pytest, pytest-cov, ruff (line-length=120), pyright (basic mode)

## Evaluation Strategy

- **Fast local eval:** `diabetes`, `blood-transfusion-service-center`, and `amazon_employee_access` (subsampled) — robust sanity check during development (`--benchmark quick`).
- **Full eval (on cluster):** TabArena benchmark (`--benchmark tabarena`). Currently runs the `nanotabpfn` subset for testing, can be switched to `classification` (all classification datasets) in `tabarena_eval.py` when running on the cluster.

## Key Design Decisions

- **Gated activations (SwiGLU, GeGLU, ReGLU)** will use identical parameter counts to non-gated variants (LLaMA-style `E → 2*(2H/3)` trick) for fair comparison.
- **Seed determinism** is critical: `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. Verified by pytest.
- Default training: 5000 steps, batch_size=32, lr=4e-3, eval every 250 steps, checkpoint every 1000 steps. Using float32.
- Model: 356K params (embedding=96, heads=3, mlp_hidden=192, layers=3, outputs=2). Heads changed to 3 to unlock FlashAttention on A100. Feature attention is chunked to avoid CUDA grid limits.

## Research Plan

1. End-to-end loop ASAP
2. Try ~15 activation functions, ~10 seeds each, everything else frozen
3. Aggregate over seeds
4. Take two best functions
5. Investigate how these behave when changing architecture (depth, etc.)
6. Evaluate on TabArena for final results + statistical significance

## Desired Final Plots

1. Pre-training steps vs. TabArena Score (baseline + 2 best variants, show std across seeds)
2. ROC-AUC per activation function (box-and-whisker)
3. Model depth vs. TabArena Score for best variants
