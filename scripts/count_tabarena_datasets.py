"""Script to count datasets in TabArena meeting various criteria."""

import pandas as pd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

ctx = TabArenaContext()
df: pd.DataFrame = ctx.task_metadata_collection.to_dataframe()  # type: ignore


def count(rows: int, feats: int, classes: int) -> int:
    """Count the number of datasets satisfying the given constraints.

    Args:
        rows: Maximum number of instances.
        feats: Maximum number of features.
        classes: Maximum number of classes.

    Returns:
        The number of unique dataset names.
    """
    subset = df[
        (df["num_instances_train"] <= rows)
        & (df["num_features"] <= feats)
        & (df["num_classes"] > 0)
        & (df["num_classes"] <= classes)
    ]
    return int(subset["dataset_name"].nunique())  # type: ignore


print("Current (3000 rows, 45 feats, 10 classes):", count(3000, 45, 10))
print("Option 1 - Super Fast (1000 rows, 20 feats, 10 classes):", count(1000, 20, 10))
print("Option 2 - Fast (2000 rows, 40 feats, 10 classes):", count(2000, 40, 10))
print("Option 3 - Medium (5000 rows, 100 feats, 10 classes):", count(5000, 100, 10))
print("Option 4 - Large (10000 rows, 100 feats, 10 classes):", count(10000, 100, 10))
print("Option 5 - Max Classification (no limits):", df[(df["num_classes"] > 0)]["dataset_name"].nunique())  # type: ignore
