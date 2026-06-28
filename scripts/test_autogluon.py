"""Test script for AutoGluon model wrapper."""

import numpy as np
import pandas as pd
import torch

import tabarena_eval
from model import NanoTabPFNModel
from tabarena_eval import TabArenaNanoTabPFNModel

device = torch.device("cpu")
model = NanoTabPFNModel(96, 4, 192, 3, 2)

tabarena_eval._CURRENT_PYTORCH_MODEL = model
tabarena_eval._CURRENT_DEVICE = device

# Create dummy data
X_train = pd.DataFrame({"cat": ["A", "B", "A", "B"] * 5, "num": np.random.randn(20)})
y_train = pd.Series([0, 1, 0, 1] * 5)

X_test = pd.DataFrame({"cat": ["A", "B", "B", "A"], "num": np.random.randn(4)})

ag_model = TabArenaNanoTabPFNModel(path="/tmp/ag_test/", name="test")
ag_model.fit(X=X_train, y=y_train)

probs = ag_model.predict_proba(X_test)
print("Predicted probabilities:")
print(probs)
