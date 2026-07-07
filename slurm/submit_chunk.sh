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

echo "====================================================="
echo "Submitting Chunk $CHUNK"
echo "Activations: ${ACTIVATIONS[*]}"
echo "====================================================="

for act in "${ACTIVATIONS[@]}"; do
  echo "-> Submitting Training Array (10 seeds) for: $act"
  # sbatch --parsable returns just the Job ID so we can capture it for the dependency
  TRAIN_ID=$(sbatch --parsable --array=0-9 --export=ALL,ACTIVATION="$act" slurm/train.sbatch)

  echo "   [Train Job ID: $TRAIN_ID]"

  echo "-> Submitting Benchmark Array (10 seeds) for: $act"
  # The benchmark array will wait peacefully in the queue until the training array succeeds
  BENCH_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" --array=0-9 --export=ALL,ACTIVATION="$act" slurm/benchmark.sbatch)

  echo "   [Benchmark Job ID: $BENCH_ID (waiting on $TRAIN_ID)]"
  echo "-----------------------------------------------------"
done

echo "Done! All jobs for chunk $CHUNK have been handed over to Slurm."
