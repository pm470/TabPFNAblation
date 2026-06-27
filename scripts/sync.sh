#!/bin/bash

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

echo "Syncing TabPFNAblation to cluster..."
echo "Destination: $REMOTE_DEST"

rsync -avz --progress \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'results' \
    --exclude '.pytest_cache' \
    --exclude '.mypy_cache' \
    --exclude '.ruff_cache' \
    --exclude '.ipynb_checkpoints' \
    --exclude '.vscode' \
    --exclude '.idea' \
    --exclude 'wandb' \
    --exclude '.agents' \
    --exclude '.env' \
    "$LOCAL_DIR" "$REMOTE_DEST"

echo "Sync complete!"
