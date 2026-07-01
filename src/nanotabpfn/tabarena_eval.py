"""TabArena evaluation module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate

from nanotabpfn.model import NanoTabPFNClassifier, NanoTabPFNModel

if TYPE_CHECKING:
    import pandas as pd
    from tabarena.utils.config_utils import ConfigGenerator


class TabArenaNanoTabPFNModel(AbstractModel):
    """TabArena wrapper for NanoTabPFN."""

    ag_key = "NanoTabPFN"
    ag_name = "NanoTabPFN"

    def __init__(self, **kwargs):
        """Initialize the wrapper."""
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """Model-specific preprocessing: label-encode categoricals, fill NaNs, to float32."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator is not None and self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0)

    def _fit(self, X: pd.DataFrame, y: pd.Series, *args, **kwargs) -> None:
        X_df = self.preprocess(X, is_train=True)
        X_np = X_df.to_numpy(dtype=np.float32)
        # NanoTabPFNClassifier expects numpy arrays
        y_np = np.asarray(y, dtype=np.int64)

        # Retrieve the shared instance (hacky but works for in-process runs)
        global _CURRENT_PYTORCH_MODEL
        global _CURRENT_DEVICE

        assert _CURRENT_PYTORCH_MODEL is not None
        assert _CURRENT_DEVICE is not None

        n_ensemble = globals().get("_CURRENT_N_ENSEMBLE", 8)
        self.model = NanoTabPFNClassifier(_CURRENT_PYTORCH_MODEL, _CURRENT_DEVICE, n_ensemble=n_ensemble)
        self.model.fit(X_np, y_np)

    def _predict_proba(self, X, **kwargs):
        """Override to ensure X is a numpy array before passing to the classifier."""
        # AutoGluon's predict_proba already calls self.preprocess, so X is preprocessed
        X_np = X.to_numpy(dtype=np.float32)
        probs = self.model.predict_proba(X_np)
        return probs

    def predict_proba(self, X, **kwargs):
        """Override to print peak memory allocated once per dataset."""
        probs = super().predict_proba(X, **kwargs)

        # Log peak memory once per dataset
        global _PRINTED_DATASETS
        try:
            path_parts = Path(self.path).parts
            base_name = self.name.split("/")[0]
            dataset_id = None
            for i, part in enumerate(path_parts):
                if part == base_name and i + 1 < len(path_parts):
                    dataset_id = path_parts[i + 1]
                    break
            if dataset_id is None and len(path_parts) >= 3:
                dataset_id = path_parts[-3]

            if dataset_id is not None and dataset_id not in _PRINTED_DATASETS:
                _PRINTED_DATASETS.add(dataset_id)
                global _CURRENT_DEVICE
                if _CURRENT_DEVICE is not None and _CURRENT_DEVICE.type == "cuda":
                    peak_mem_gb = torch.cuda.max_memory_allocated(_CURRENT_DEVICE) / (1024**3)
                    print(f"[NanoTabPFN] Peak GPU memory allocated for dataset {dataset_id}: {peak_mem_gb:.2f} GB")
        except Exception:
            pass

        return probs

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        """Return supported problem types."""
        return ["binary", "multiclass"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        """Return config generator."""
        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(
            model_cls=cls,
            manual_configs=[{}],
            search_space={},
        )


# Global variables for in-process sharing
_CURRENT_PYTORCH_MODEL: NanoTabPFNModel | None = None
_CURRENT_DEVICE: torch.device | None = None
_CURRENT_N_ENSEMBLE: int = 8
_PRINTED_DATASETS: set[str] = set()


def run_tabarena_eval(
    model: NanoTabPFNModel, device: torch.device, run_dir: Path, subset: str = "classification", n_ensemble: int = 8
):
    """Run TabArena evaluation using the provided trained model."""
    global _CURRENT_PYTORCH_MODEL
    global _CURRENT_DEVICE
    global _CURRENT_N_ENSEMBLE

    _CURRENT_PYTORCH_MODEL = model
    _CURRENT_DEVICE = device
    _CURRENT_N_ENSEMBLE = n_ensemble

    results_dir = str(run_dir / "tabarena_exp")

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            (TabArenaNanoTabPFNModel.config_generator(), 0),
        ],
    ).build_experiments()

    TabArenaContext.SUBSET_PREDICATES["nanotabpfn"] = SubsetPredicate(
        lambda df: (
            (df["max_train_rows"] <= 10_000)
            & (df["n_features"] <= 500)
            & (df["n_classes"] > 0)
            & (df["n_classes"] <= 10)
        ),
        ("max_train_rows", "n_features", "n_classes"),
    )

    context = TabArenaContext()

    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=subset,
        new_result_prefix="[New] ",
        debug_mode=True,  # In-process debugging required for our global variable hack
    )

    print(f"TabArena evaluation completed. Results in {results_dir}")
