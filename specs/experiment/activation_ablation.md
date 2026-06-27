---
id: EXP-001
title: "Activation Ablation Experiment Runner"
status: implemented
module: run_experiment.py
last_synced: 2026-06-28
---

# Activation Ablation Experiment Runner

## User Story

As a researcher, I want a CLI tool that runs a single ablation experiment with a specified activation function and seed, logging all configuration, metrics, and checkpoints to a structured output directory, so that I can systematically compare activation function variants.

## Acceptance Criteria

### CLI Interface (parse_args)

- [x] AC-1: `--activation` (str, default `"gelu"`) — activation function name.
- [x] AC-2: `--benchmark` (str, choices `["breast_cancer", "tabarena"]`, default `"breast_cancer"`) — benchmark for final evaluation.
- [x] AC-3: `--seed` (int, default `0`) — random seed.
- [x] AC-4: `--num_steps` (int, default `2500`) — number of training steps.
- [x] AC-5: `--batch_size` (int, default `32`) — batch size.
- [x] AC-6: `--lr` (float, default `4e-3`) — learning rate.
- [x] AC-7: `--eval_every` (int, default `25`) — evaluation interval in steps.
- [x] AC-8: `--checkpoint_every` (int, default `250`) — checkpoint save interval in steps.
- [x] AC-9: `--output_dir` (str, default `"results"`) — base output directory.
- [x] AC-10: Dataset args: `--max_seq_len` (1000), `--max_features` (60), `--max_classes` (10).
- [x] AC-11: Model architecture args: `--embedding_size` (96), `--num_attention_heads` (4), `--mlp_hidden_size` (192), `--num_layers` (3), `--num_outputs` (10).

### Output Directory Structure

- [x] AC-12: Run directory is `{output_dir}/{activation}/seed_{seed}/`.
- [x] AC-13: Run directory is created with `parents=True, exist_ok=True`.
- [x] AC-14: Checkpoint directory is `{run_dir}/checkpoints/`.

### Configuration Saving

- [x] AC-15: `config.json` is saved in the run directory before training starts.
- [x] AC-16: Config contains all CLI args plus computed `param_count` and `device`.
- [x] AC-17: Config keys include: `activation`, `benchmark`, `seed`, `num_steps`, `batch_size`, `lr`, `eval_every`, `checkpoint_every`, `embedding_size`, `num_attention_heads`, `mlp_hidden_size`, `num_layers`, `num_outputs`, `max_seq_len`, `max_features`, `max_classes`, `param_count`, `device`.
- [x] AC-18: Config is written with `json.dump` using `indent=2`.

### Training

- [x] AC-19: `set_randomness_seed(args.seed)` is called before model creation.
- [x] AC-20: Passes `eval` (from train.py) as the `eval_func` to `train()`.
- [x] AC-21: Passes `checkpoint_dir` and `checkpoint_every` from CLI args to `train()`.

### Metrics Logging

- [x] AC-22: `metrics.jsonl` is saved in the run directory after training completes.
- [x] AC-23: Each line is a JSON object written via `json.dumps(entry) + "\n"`.
- [x] AC-24: Each entry contains `step`, `wall_time`, `loss`, `roc_auc`, `acc`, `balanced_acc` (when eval_func is provided).

### Final Evaluation

- [x] AC-25: Model is set to `.eval()` mode before final evaluation.
- [x] AC-26: If `--benchmark tabarena`, imports and calls `run_tabarena_eval(model, device, run_dir)`.
- [x] AC-27: If `--benchmark breast_cancer`, creates a `NanoTabPFNClassifier`, runs `eval()`, and saves results to `final_scores.json`.
- [x] AC-28: `final_scores.json` is written with `json.dump` using `indent=2`.

### Return Value

- [x] AC-29: `run_experiment()` returns the `eval_history` list.

## Notes

- The `--activation` flag is currently a placeholder — it is saved in config but does not yet alter the model architecture. The model always uses GELU.
- `parse_args` accepts an optional `argv` parameter for testability.
- The `param_count` in config is the total number of trainable parameters, computed as `sum(p.numel() for p in model.parameters())`.
