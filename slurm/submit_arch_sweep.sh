#!/bin/bash
# Submits architecture ablation sweep jobs to Slurm.
# Usage: bash slurm/submit_arch_sweep.sh [--dry-run] <sweep_axis>
#        sweep_axis: layers | hidden | embedding | all

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" == "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

AXIS="${1:-}"
if [ -z "$AXIS" ]; then
  echo "Usage: $0 [--dry-run] <layers|hidden|embedding|all>"
  exit 1
fi

# ─── Sweep grid definitions ───
ACTIVATIONS=("gelu" "swiglu" "bilinear")
SEEDS=(0 1 2)

# Baseline: E=128, H=256, L=3
BASELINE_E=128
BASELINE_H=256
BASELINE_L=3

# Per-axis sweep values
LAYER_VALUES=(1 2 3 4 6)
HIDDEN_VALUES=(64 128 256 384 512)
EMBEDDING_VALUES=(64 128 192 256)

# ─── Load environment ───
if [ -f .env ]; then
  source .env
fi
WORKSPACE_DIR="${WORKSPACE_DIR:-/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data}"
RESULTS_DIR="${WORKSPACE_DIR}/results_arch"

# ─── Helper: submit one (activation, E, H, L, axis) combo ───
submit_config() {
  local act="$1"
  local emb="$2"
  local hidden="$3"
  local layers="$4"
  local axis="$5"

  local arch_tag="e${emb}_h${hidden}_l${layers}"
  local needed_seeds=""

  for seed in "${SEEDS[@]}"; do
    local final="${RESULTS_DIR}/${axis}/${act}/${arch_tag}/seed_${seed}/checkpoints/final.pt"
    if [ ! -f "$final" ]; then
      needed_seeds="${needed_seeds}${needed_seeds:+,}${seed}"
    fi
  done

  if [ -z "$needed_seeds" ]; then
    echo "   [${act} ${arch_tag}] All seeds complete. Skipping."
    return
  fi

  local cmd="sbatch --parsable --array=\"${needed_seeds}\" --export=ALL,ACTIVATION=\"${act}\",SWEEP_AXIS=\"${axis}\",EMBEDDING_SIZE=\"${emb}\",MLP_HIDDEN_SIZE=\"${hidden}\",NUM_LAYERS=\"${layers}\" slurm/train_arch.sbatch"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "   [DRY-RUN] ${cmd}"
  else
    local job_id
    job_id=$(sbatch --parsable --array="${needed_seeds}" \
      --export=ALL,ACTIVATION="${act}",SWEEP_AXIS="${axis}",EMBEDDING_SIZE="${emb}",MLP_HIDDEN_SIZE="${hidden}",NUM_LAYERS="${layers}" \
      slurm/train_arch.sbatch)
    echo "   [${act} ${arch_tag}] Job ${job_id} (seeds: ${needed_seeds})"
  fi
}

# ─── Sweep submission ───
submit_axis() {
  local axis="$1"
  shift
  local -a values=("$@")

  echo ""
  echo "====================================================="
  echo "Sweep axis: ${axis}"
  echo "====================================================="

  for act in "${ACTIVATIONS[@]}"; do
    echo "-> ${act}"
    for val in "${values[@]}"; do
      case "$axis" in
        layers)
          submit_config "$act" "$BASELINE_E" "$BASELINE_H" "$val" "$axis"
          ;;
        hidden)
          submit_config "$act" "$BASELINE_E" "$val" "$BASELINE_L" "$axis"
          ;;
        embedding)
          submit_config "$act" "$val" "$BASELINE_H" "$BASELINE_L" "$axis"
          ;;
      esac
    done
    echo "-----------------------------------------------------"
  done
}

# ─── Main dispatch ───
case "$AXIS" in
  layers)
    submit_axis "layers" "${LAYER_VALUES[@]}"
    ;;
  hidden)
    submit_axis "hidden" "${HIDDEN_VALUES[@]}"
    ;;
  embedding)
    submit_axis "embedding" "${EMBEDDING_VALUES[@]}"
    ;;
  all)
    submit_axis "layers" "${LAYER_VALUES[@]}"
    submit_axis "hidden" "${HIDDEN_VALUES[@]}"
    submit_axis "embedding" "${EMBEDDING_VALUES[@]}"
    ;;
  *)
    echo "Error: Invalid axis '${AXIS}'. Use: layers, hidden, embedding, or all"
    exit 1
    ;;
esac

echo ""
echo "Done! All necessary jobs for axis '${AXIS}' have been submitted."
