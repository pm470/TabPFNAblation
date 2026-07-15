"""Script to count datasets and report actual max dimensions in TabArena subsets."""

import pandas as pd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

ctx = TabArenaContext()
df: pd.DataFrame = ctx.task_metadata_collection.to_dataframe()  # type: ignore


def analyze(rows: int, feats: int, classes: int, label: str) -> None:
    """Count datasets and print actual max dimensions for the given constraints.

    Args:
        rows: Maximum number of training instances.
        feats: Maximum number of features.
        classes: Maximum number of classes.
        label: Human-readable label for this configuration.
    """
    subset = df[
        (df["num_instances_train"] <= rows)
        & (df["num_features"] <= feats)
        & (df["num_classes"] > 0)
        & (df["num_classes"] <= classes)
    ]
    n_datasets = int(subset["dataset_name"].nunique())  # type: ignore
    if n_datasets == 0:
        print(f"{label}: 0 datasets")
        return

    max_train_rows = int(subset["num_instances_train"].max())
    max_total_rows = int(subset["num_instances"].max())
    max_feats = int(subset["num_features"].max())
    max_cls = int(subset["num_classes"].max())
    print(
        f"{label}: {n_datasets} datasets | "
        f"actual max: {max_train_rows} train rows ({max_total_rows} total), "
        f"{max_feats} features, {max_cls} classes"
    )


print("=== TabArena Classification Subset Analysis ===\n")

analyze(3000, 45, 10, "Old nanotabpfn (3000 rows, 45 feats, 10 classes)")
analyze(10000, 120, 10, "New nanotabpfn (10000 rows, 120 feats, 10 classes)")
analyze(10000, 500, 10, "tabpfn predicate (10000 rows, 500 feats, 10 classes)")

print("\n--- Exploration options ---")
analyze(1000, 20, 10, "Super Fast (1000 rows, 20 feats, 10 classes)")
analyze(2000, 40, 10, "Fast (2000 rows, 40 feats, 10 classes)")
analyze(5000, 100, 10, "Medium (5000 rows, 100 feats, 10 classes)")
analyze(10000, 100, 10, "Large (10000 rows, 100 feats, 10 classes)")

n_all_clf = df[(df["num_classes"] > 0)]["dataset_name"].nunique()  # type: ignore
max_all_rows = int(df[df["num_classes"] > 0]["num_instances_train"].max())
max_all_feats = int(df[df["num_classes"] > 0]["num_features"].max())
max_all_cls = int(df[df["num_classes"] > 0]["num_classes"].max())
print(
    f"All classification (no limits): {n_all_clf} datasets | "
    f"actual max: {max_all_rows} train rows, {max_all_feats} features, {max_all_cls} classes"
)
