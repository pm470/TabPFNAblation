from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import json

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

from model import NanoTabPFNClassifier, NanoTabPFNModel

if TYPE_CHECKING:
    import pandas as pd
    from tabarena.utils.config_utils import ConfigGenerator


class TabArenaNanoTabPFNModel(AbstractModel):
    ag_key = "NanoTabPFN"
    ag_name = "NanoTabPFN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> np.ndarray:
        """Model-specific preprocessing: label-encode categoricals, fill NaNs, to float32."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, **kwargs) -> None:
        X = self.preprocess(X, y=y, is_train=True)
        # NanoTabPFNClassifier expects numpy arrays
        y_np = y.to_numpy(dtype=np.int64) if hasattr(y, "to_numpy") else y

        # Retrieve the shared instance (hacky but works for in-process runs)
        global _CURRENT_PYTORCH_MODEL
        global _CURRENT_DEVICE
        
        self.model = NanoTabPFNClassifier(_CURRENT_PYTORCH_MODEL, _CURRENT_DEVICE)
        self.model.fit(X, y_np)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        from tabarena.utils.config_utils import ConfigGenerator
        return ConfigGenerator(
            model_cls=cls,
            manual_configs=[{}],
            search_space={},
        )

# Global variables for in-process sharing
_CURRENT_PYTORCH_MODEL = None
_CURRENT_DEVICE = None

def run_tabarena_eval(model: NanoTabPFNModel, device: torch.device, run_dir: Path):
    """
    Run TabArena evaluation using the provided trained model.
    """
    global _CURRENT_PYTORCH_MODEL
    global _CURRENT_DEVICE
    
    _CURRENT_PYTORCH_MODEL = model
    _CURRENT_DEVICE = device
    
    results_dir = str(run_dir / "tabarena_exp")

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            (TabArenaNanoTabPFNModel.config_generator(), 0),
        ],
    ).build_experiments()

    context = TabArenaContext()
    
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset="full",
        build_kwargs={"dataset_names": ["blood-transfusion-service-center"]},  # test on a single classification dataset
        new_result_prefix="[New] ",
        debug_mode=True,  # In-process debugging required for our global variable hack
    )
    
    print(f"TabArena evaluation completed. Results in {results_dir}")
