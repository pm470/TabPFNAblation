#!/bin/bash
# Submits a chunk of the ablation study to Slurm.
# Designed to be split across 3 or 4 cluster accounts to maximize fair-share priority.
# Usage: bash slurm/submit_chunk.sh [--dry-run] <chunk_number>

DRY_RUN=0
if [ "$1" == "--dry-run" ]; then
  DRY_RUN=1
  shift
elif [ "$2" == "--dry-run" ]; then
  DRY_RUN=1
fi

if [ -z "$1" ] || [ "$1" == "--dry-run" ]; then
  echo "Usage: $0 [--dry-run] <chunk_number> (1, 2, 3, or 4)"
  exit 1
fi

CHUNK=$1

# Define the activations split into fair-share chunks
case $CHUNK in
1)
  ACTIVATIONS=("gelu" "relu" "swish")
  GATED_UNRESTRICTED=0
  ;;
2)
  ACTIVATIONS=("prelu" "leaky_relu")
  GATED_UNRESTRICTED=0
  ;;
3)
  ACTIVATIONS=("swiglu" "bilinear")
  GATED_UNRESTRICTED=0
  ;;
4)
  ACTIVATIONS=("swiglu" "bilinear")
  GATED_UNRESTRICTED=1
  ;;
*)
  echo "Error: Invalid chunk number. Please use 1, 2, 3, or 4."
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
echo "Gated Unrestricted: $GATED_UNRESTRICTED"
echo "====================================================="

for act in "${ACTIVATIONS[@]}"; do
  TRAIN_SEEDS=""
  BENCH_SEEDS=""
  
  DIR_NAME="$act"
  if [ "$GATED_UNRESTRICTED" -eq 1 ]; then
    DIR_NAME="${act}_unrestricted"
  fi

  for seed in {0..19}; do
    # Check if training is completed
    if [ ! -f "${RESULTS_DIR}/${DIR_NAME}/seed_${seed}/checkpoints/final.pt" ]; then
      TRAIN_SEEDS="${TRAIN_SEEDS}${TRAIN_SEEDS:+,}${seed}"
    fi
    # Check if benchmarking is completed
    if [ ! -f "${RESULTS_DIR}/${DIR_NAME}/seed_${seed}/benchmark_final/tabarena_exp/nanotabpfn_summary.csv" ]; then
      BENCH_SEEDS="${BENCH_SEEDS}${BENCH_SEEDS:+,}${seed}"
    fi
  done

  echo "-> Status for: $act (dir=$DIR_NAME)"

  if [ -z "$TRAIN_SEEDS" ]; then
    echo "   [Training fully complete. Skipping train job.]"
    TRAIN_ID=""
  else
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "   [DRY-RUN] sbatch --parsable --array=\"$TRAIN_SEEDS\" --export=ALL,ACTIVATION=\"$act\",GATED_UNRESTRICTED=\"$GATED_UNRESTRICTED\" slurm/train.sbatch"
      TRAIN_ID="DRY_RUN_TRAIN_ID"
    else
      TRAIN_ID=$(sbatch --parsable --array="$TRAIN_SEEDS" --export=ALL,ACTIVATION="$act",GATED_UNRESTRICTED="$GATED_UNRESTRICTED" slurm/train.sbatch)
    fi
    echo "   [Train Job ID: $TRAIN_ID (Queued Seeds: $TRAIN_SEEDS)]"
  fi

  if [ -z "$BENCH_SEEDS" ]; then
    echo "   [Benchmarking fully complete. Skipping bench job.]"
  else
    if [ -n "$TRAIN_ID" ]; then
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "   [DRY-RUN] sbatch --parsable --dependency=afterok:\"$TRAIN_ID\" --array=\"$BENCH_SEEDS\" --export=ALL,ACTIVATION=\"$act\",GATED_UNRESTRICTED=\"$GATED_UNRESTRICTED\" slurm/benchmark.sbatch"
        BENCH_ID="DRY_RUN_BENCH_ID"
      else
        BENCH_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" --array="$BENCH_SEEDS" --export=ALL,ACTIVATION="$act",GATED_UNRESTRICTED="$GATED_UNRESTRICTED" slurm/benchmark.sbatch)
      fi
      echo "   [Benchmark Job ID: $BENCH_ID (Queued Seeds: $BENCH_SEEDS, waiting on $TRAIN_ID)]"
    else
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "   [DRY-RUN] sbatch --parsable --array=\"$BENCH_SEEDS\" --export=ALL,ACTIVATION=\"$act\",GATED_UNRESTRICTED=\"$GATED_UNRESTRICTED\" slurm/benchmark.sbatch"
        BENCH_ID="DRY_RUN_BENCH_ID"
      else
        BENCH_ID=$(sbatch --parsable --array="$BENCH_SEEDS" --export=ALL,ACTIVATION="$act",GATED_UNRESTRICTED="$GATED_UNRESTRICTED" slurm/benchmark.sbatch)
      fi
      echo "   [Benchmark Job ID: $BENCH_ID (Queued Seeds: $BENCH_SEEDS, starting immediately)]"
    fi
  fi
  echo "-----------------------------------------------------"
done

echo "Done! All necessary jobs for chunk $CHUNK have been handed over to Slurm."

