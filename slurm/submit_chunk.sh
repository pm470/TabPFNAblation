#!/bin/bash
# Submits a chunk of the ablation study to Slurm.
# Designed to be split across 3 cluster accounts to maximize fair-share priority.
# Usage: bash slurm/submit_chunk.sh <chunk_number>

if [ -z "$1" ]; then
  echo "Usage: $0 <chunk_number> (1, 2, or 3)"
  exit 1
fi

CHUNK=$1

# Define the 15 activations split into 3 fair-share chunks
case $CHUNK in
1)
  # TODO: Replace with the first 5 functions (e.g., "gelu" "swiglu" ...)
  ACTIVATIONS=("func1" "func2" "func3" "func4" "func5")
  ;;
2)
  # TODO: Replace with the next 5 functions
  ACTIVATIONS=("func6" "func7" "func8" "func9" "func10")
  ;;
3)
  # TODO: Replace with the final 5 functions
  ACTIVATIONS=("func11" "func12" "func13" "func14" "func15")
  ;;
*)
  echo "Error: Invalid chunk number. Please use 1, 2, or 3."
  exit 1
  ;;
esac

if [ -f .env ]; then
  source .env
fi
WORKSPACE_DIR="${WORKSPACE_DIR:-/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data}"
RESULTS_DIR="${WORKSPACE_DIR}/results"

echo "====================================================="
echo "Submitting Chunk $CHUNK"
echo "Activations: ${ACTIVATIONS[*]}"
echo "====================================================="

for act in "${ACTIVATIONS[@]}"; do
  TRAIN_SEEDS=""
  BENCH_SEEDS=""
  
  for seed in {0..9}; do
    # Check if training is completed
    if [ ! -f "${RESULTS_DIR}/${act}/seed_${seed}/checkpoints/final.pt" ]; then
      TRAIN_SEEDS="${TRAIN_SEEDS}${TRAIN_SEEDS:+,}${seed}"
    fi
    # Check if benchmarking is completed
    if [ ! -f "${RESULTS_DIR}/${act}/seed_${seed}/tabarena_exp/nanotabpfn_summary.csv" ]; then
      BENCH_SEEDS="${BENCH_SEEDS}${BENCH_SEEDS:+,}${seed}"
    fi
  done

  echo "-> Status for: $act"

  if [ -z "$TRAIN_SEEDS" ]; then
    echo "   [Training fully complete. Skipping train job.]"
    TRAIN_ID=""
  else
    TRAIN_ID=$(sbatch --parsable --array="$TRAIN_SEEDS" --export=ALL,ACTIVATION="$act" slurm/train.sbatch)
    echo "   [Train Job ID: $TRAIN_ID (Queued Seeds: $TRAIN_SEEDS)]"
  fi

  if [ -z "$BENCH_SEEDS" ]; then
    echo "   [Benchmarking fully complete. Skipping bench job.]"
  else
    if [ -n "$TRAIN_ID" ]; then
      BENCH_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" --array="$BENCH_SEEDS" --export=ALL,ACTIVATION="$act" slurm/benchmark.sbatch)
      echo "   [Benchmark Job ID: $BENCH_ID (Queued Seeds: $BENCH_SEEDS, waiting on $TRAIN_ID)]"
    else
      BENCH_ID=$(sbatch --parsable --array="$BENCH_SEEDS" --export=ALL,ACTIVATION="$act" slurm/benchmark.sbatch)
      echo "   [Benchmark Job ID: $BENCH_ID (Queued Seeds: $BENCH_SEEDS, starting immediately)]"
    fi
  fi
  echo "-----------------------------------------------------"
done

echo "Done! All necessary jobs for chunk $CHUNK have been handed over to Slurm."
