#!/usr/bin/env bash
# Run all activation × seed combinations sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTIVATIONS=("bilinear" "swiglu" "relu" "gelu" "swish" "prelu" "leaky_relu")
SEEDS=(0 1 2)
NUM_STEPS=2500
EVAL_EVERY=25
CHECKPOINT_EVERY=250

echo "=== TabPFN Ablation Study ==="
echo "Activations: ${ACTIVATIONS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Steps: ${NUM_STEPS}"
echo ""

for activation in "${ACTIVATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "--- Running: activation=${activation}, seed=${seed} ---"
        python run_experiment.py \
            --activation "$activation" \
            --seed "$seed" \
            --num_steps "$NUM_STEPS" \
            --eval_every "$EVAL_EVERY" \
            --checkpoint_every "$CHECKPOINT_EVERY"
        echo ""
    done
done

echo "=== All runs complete ==="
echo "Results in: results/"
echo "Run 'python plot_results.py' to generate plots."
