from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_eval_datasets():
    """Mock get_eval_datasets to return a small toy dataset to speed up unit tests."""
    toy_datasets = [
        (
            "toy_diabetes",
            np.random.randn(10, 5).astype(np.float32),
            np.random.randn(5, 5).astype(np.float32),
            np.random.randint(0, 2, (10,)),
            np.random.randint(0, 2, (5,)),
        )
    ]
    with patch("nanotabpfn.train.get_eval_datasets", return_value=toy_datasets):
        yield
