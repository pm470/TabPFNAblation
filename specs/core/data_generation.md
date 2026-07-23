---
id: CORE-003
title: "Data Generation"
status: implemented
module: scripts/data/generate_prior.py
last_synced: 2026-07-01
---

# Data Generation

## User Story

As a researcher, I want to generate a synthetic prior dataset using `nanoTabICL` prior and save it to an HDF5 file, so that I can pre-train my TabPFN models with large enough contexts and dimensionalities to perform well on TabArena.

## Acceptance Criteria

### Data Generation Script (`scripts/data/generate_prior.py`)

- [x] AC-1: The script accepts `--output_file`, `--num_datasets`, `--batch_size`, `--max_seq_len`, `--max_features`, `--max_classes`, and `--num_workers` as CLI arguments.
- [x] AC-2: Uses `NanopriorDataset` from `train.py` to generate the data on the fly.
- [x] AC-3: Uses a PyTorch `DataLoader` with the specified `num_workers` to parallelize data generation.
- [x] AC-4: Creates an HDF5 file with the required datasets: `max_num_classes`, `num_features`, `num_datapoints`, `single_eval_pos`, `X`, and `y`.
- [x] AC-5: `X` is stored as a 3D array of shape `(num_datasets, max_seq_len, max_features)`.
- [x] AC-6: `y` is stored as a 2D array of shape `(num_datasets, max_seq_len)`.
- [x] AC-7: `max_num_classes` is saved as a single scalar or 1D array of length 1.
- [x] AC-8: `num_features`, `num_datapoints`, and `single_eval_pos` are 1D arrays of length `num_datasets`.
- [x] AC-9: Incrementally writes batches to the HDF5 file to prevent out-of-memory errors on the CPU.
- [x] AC-10: Saves the actual generated shape (`n_samples` and `n_features`) for each batch into `num_datapoints` and `num_features` respectively.
