#!/bin/bash

# Define local and remote paths
LOCAL_DIR="$HOME/code/TabPFNAblation/"
REMOTE_DEST="fr_lf453@uc3.scc.kit.edu:c/"

echo "Syncing TabPFNAblation to cluster..."

rsync -avz \
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
  "$LOCAL_DIR" "$REMOTE_DEST"

echo "Sync complete!"
