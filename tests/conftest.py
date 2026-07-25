from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="session")
def mock_eval_datasets():
    """Mock get_eval_datasets to return a small toy dataset to speed up unit tests."""
    toy_datasets = [
        (
            "toy_diabetes",
            np.random.randn(10, 5).astype(np.float32),
            np.random.randn(6, 5).astype(np.float32),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
            np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        )
    ]
    with patch("nanotabpfn.train.get_eval_datasets", return_value=toy_datasets):
        yield
