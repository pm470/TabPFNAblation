"""TabArena evaluation module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

from nanotabpfn.model import NanoTabPFNClassifier, NanoTabPFNModel
from nanotabpfn.utils import save_memory_stat

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
        """Preprocess and predict.

        AutoGluon does NOT automatically call self.preprocess before _predict_proba.
        The BAG model passes preprocess_nonadaptive=False via **kwargs so only the
        stateful steps run (including _preprocess_align_features, which aligns features
        to the training fold — critical when drop_unique removed a single-value feature).
        """
        X = self.preprocess(X, **kwargs)
        X_np = X.to_numpy(dtype=np.float32)
        probs = self.model.predict_proba(X_np)
        return self._convert_proba_to_unified_form(probs)

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
                    _PEAK_MEM_BY_DATASET_GB[dataset_id] = peak_mem_gb
        except Exception:
            pass

        return probs

    def score_with_y_pred_proba(self, y, y_pred_proba, **kwargs):
        """Override to handle datasets where one fold has only a single class, which crashes ROC AUC."""
        try:
            return super().score_with_y_pred_proba(y, y_pred_proba, **kwargs)
        except ValueError as e:
            if "Only one class present in y_true" in str(e):
                return 0.5
            raise

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
_PEAK_MEM_BY_DATASET_GB: dict[str, float] = {}


def run_tabarena_eval(
    model: NanoTabPFNModel, device: torch.device, run_dir: Path, subset: str = "nanotabpfn", n_ensemble: int = 8
):
    """Run TabArena evaluation using the provided trained model."""
    global _CURRENT_PYTORCH_MODEL
    global _CURRENT_DEVICE
    global _CURRENT_N_ENSEMBLE

    _CURRENT_PYTORCH_MODEL = model
    _CURRENT_DEVICE = device
    _CURRENT_N_ENSEMBLE = n_ensemble

    # Reset per-run memory tracking state so stale entries from a previous
    # run_tabarena_eval call (e.g. in the same process/test) don't leak in.
    _PRINTED_DATASETS.clear()
    _PEAK_MEM_BY_DATASET_GB.clear()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results_dir = str(run_dir / "tabarena_exp")

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            (TabArenaNanoTabPFNModel.config_generator(), 0),
        ],
        verbosity=0,
        model_verbosity=0,
    ).build_experiments()

    tabpfn_obj = TabArenaContext.SUBSET_PREDICATES["tabpfn"]
    tabpfn_pred = tabpfn_obj.predicate
    req_cols = tuple(set((*tabpfn_obj.required_columns, "n_classes")))

    TabArenaContext.SUBSET_PREDICATES["nanotabpfn"] = SubsetPredicate(
        lambda df: tabpfn_pred(df) & (df["n_classes"] > 0),
        req_cols,
    )

    context = TabArenaContext()

    job_results = context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=subset,
        new_result_prefix="[New] ",
        debug_mode=True,  # In-process debugging required for our global variable hack
    )

    if device.type == "cuda" and _PEAK_MEM_BY_DATASET_GB:
        save_memory_stat(run_dir, "peak_vram_eval_gb", max(_PEAK_MEM_BY_DATASET_GB.values()))
        save_memory_stat(run_dir, "peak_vram_eval_by_dataset_gb", dict(_PEAK_MEM_BY_DATASET_GB))

    print("Generating benchmark summary...")
    try:
        import pandas as pd
        from sklearn.metrics import log_loss, roc_auc_score

        records = []

        for res in job_results:
            if isinstance(res, dict):
                task_id = res.get("task_metadata", {}).get("tid", "unknown")
                fold = res.get("task_metadata", {}).get("fold", "unknown")
                framework = res.get("framework")

                roc_auc = None
                loss = None

                # Extract from simulation_artifacts if available
                sim_artifacts = res.get("simulation_artifacts", {})

                # Compute metrics directly from raw predictions if available
                if "y_test" in sim_artifacts and "pred_proba_dict_test" in sim_artifacts:
                    y_true = sim_artifacts["y_test"]
                    y_pred = sim_artifacts["pred_proba_dict_test"].get(framework)

                    if y_pred is not None:
                        # 1. Log Loss
                        import contextlib

                        with contextlib.suppress(Exception):
                            loss = log_loss(y_true, y_pred, labels=list(range(y_pred.shape[1])))

                        # 2. ROC AUC
                        try:
                            if res.get("problem_type") == "binary":
                                if y_pred.ndim == 2:
                                    roc_auc = roc_auc_score(y_true, y_pred[:, 1])
                                else:
                                    roc_auc = roc_auc_score(y_true, y_pred)
                            else:
                                # For multiclass, explicitly provide labels to prevent errors if a fold misses a class
                                labels = list(range(y_pred.shape[1]))
                                roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovr", labels=labels)
                        except Exception as e:
                            print(f"ROC_AUC failed for {task_id}: {e}")

                # Fallback to metric_error if manual calculation fails
                if loss is None:
                    loss = res.get("metric_error")

                if loss is not None or roc_auc is not None:
                    records.append({"task_id": task_id, "fold": fold, "roc_auc": roc_auc, "log_loss": loss})

        if not records:
            print("\nWarning: Could not extract metric scores directly from the job_results list.")
            print("Job results sample keys:", list(job_results[0].keys()) if job_results else "Empty")
        else:
            df = pd.DataFrame(records)
            print(f"\nTabArena evaluation completed. Results in {results_dir}")
            print("\nFinal Mean Scores (over all datasets & folds):")

            # Print mean of numeric columns only
            mean_scores = df.mean(numeric_only=True)
            print(mean_scores.to_string())  # type: ignore

            # Save to CSV
            out_csv = Path(results_dir) / "nanotabpfn_summary.csv"
            df.groupby("task_id").mean(numeric_only=True).to_csv(out_csv)
            print(f"\nSaved per-dataset summary to: {out_csv}")

    except Exception as e:
        print(f"\nTabArena evaluation completed. Results in {results_dir}")
        print(f"Failed to generate summary: {e}")
