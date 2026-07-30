#!/bin/bash
# Pulls selectively (only metrics, plots, and csvs) from the cluster to local

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Define local path
LOCAL_DIR="${LOCAL_DIR:-$HOME/code/TabPFNAblation/}"

# Define remote connection details (fallback to defaults if not in .env)
CLUSTER_USER="${CLUSTER_USER:-fr_lf453}"
CLUSTER_HOST="${CLUSTER_HOST:-uc3.scc.kit.edu}"
CLUSTER_DIR="${CLUSTER_DIR:-TabPFNAblation/}"

# Construct REMOTE_DEST if not explicitly provided
REMOTE_DEST="${REMOTE_DEST:-${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DIR}}"

echo "Pulling results from cluster..."
echo "Source: $REMOTE_DEST"

# Safe Sync Logic:
# 1. Include all directories so we can search inside them
# 2. Include ONLY .png/.svg (plots), .jsonl (fast eval metrics), and .csv (TabArena results)
# 3. Exclude EVERYTHING else (no .py files, no massive .pt checkpoints, no data)
rsync -avz --progress \
    --exclude=".venv" \
    --exclude=".git" \
    --exclude=".ruff_cache" \
    --exclude=".pytest_cache" \
    --include="*/" \
    --include="*.png" \
    --include="*.svg" \
    --include="*.csv" \
    --include="*.jsonl" \
    --exclude="*" \
    "$REMOTE_DEST" "$LOCAL_DIR"

echo "Results pulled safely!"
