"""Training loops and data loading."""

import os
import random
import time

import h5py
import numpy as np
import schedulefree
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from nanotabpfn.model import NanoTabPFNClassifier, NanoTabPFNModel


def set_randomness_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_device() -> torch.device:
    """Get default torch device."""
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return torch.device(device)


def get_eval_datasets():
    """Returns a list of (X_train, X_test, y_train, y_test) tuples for evaluation."""
    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    datasets = []

    def fetch_and_prep(name, subsample_n=None):
        try:
            X, y = fetch_openml(name, version=1, parser="auto", return_X_y=True)

            # Simple conversion for ablation eval
            if isinstance(X, pd.DataFrame):
                X = X.copy()
                for col in X.select_dtypes(include=['category', 'object']).columns:
                    X[col] = X[col].astype('category').cat.codes
                X = X.fillna(0).values.astype(np.float32)

            y = np.array(LabelEncoder().fit_transform(y), dtype=np.int64)

            if subsample_n and len(X) > subsample_n:
                _, X, _, y = train_test_split(X, y, test_size=subsample_n, stratify=y, random_state=42)

            return train_test_split(X, y, test_size=0.5, random_state=42)
        except Exception as e:
            print(f"Failed to fetch/prep {name}: {e}")
            return None

    for d in ["diabetes", "blood-transfusion-service-center"]:
        res = fetch_and_prep(d)
        if res is not None:
            datasets.append((d, *res))

    amazon = fetch_and_prep("amazon_employee_access", subsample_n=2000)
    if amazon is not None:
        datasets.append(("amazon_employee_access", *amazon))

    return datasets


def eval(classifier, datasets=None):
    """Evaluate classifier on datasets."""
    if datasets is None:
        datasets = get_eval_datasets()
    scores: dict = {"roc_auc": 0.0, "acc": 0.0, "balanced_acc": 0.0, "datasets": {}}
    for name, X_train, X_test, y_train, y_test in datasets:
        classifier.fit(X_train, y_train)
        prob = classifier.predict_proba(X_test)
        if np.isnan(prob).any():
            print("Warning: NaN predictions detected during eval. Replacing with uniform probabilities.")
            prob = np.nan_to_num(prob, nan=1.0 / prob.shape[1])
        pred = prob.argmax(axis=1)  # avoid a second forward pass by not calling predict
        
        if prob.shape[1] == 2:
            prob = prob[:, 1]
            ds_roc_auc = float(roc_auc_score(y_test, prob, multi_class="ovr"))
        else:
            ds_roc_auc = float(roc_auc_score(y_test, prob, multi_class="ovr", labels=np.arange(prob.shape[1])))
            
        ds_acc = float(accuracy_score(y_test, pred))
        ds_bal_acc = float(balanced_accuracy_score(y_test, pred))
        
        scores["datasets"][name] = {
            "roc_auc": ds_roc_auc,
            "acc": ds_acc,
            "balanced_acc": ds_bal_acc
        }
        
        scores["roc_auc"] += ds_roc_auc
        scores["acc"] += ds_acc
        scores["balanced_acc"] += ds_bal_acc

    scores["roc_auc"] /= len(datasets)
    scores["acc"] /= len(datasets)
    scores["balanced_acc"] /= len(datasets)
    return scores


def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    lr: float = 1e-4,
    device: torch.device | None = None,
    steps_per_eval=10,
    eval_func=None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int | None = None,
    checkpoint_every_minutes: float | None = None,
    start_step: int = 0,
):
    """Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        lr: (float) learning rate
        device: (torch.device) the device we are using
        steps_per_eval: (int) how many steps we wait before running evaluation again
        eval_func: a function that takes in a classifier and returns a dict containing the average scores
                   for some metrics and datasets
        checkpoint_dir: (str|None) directory to save model checkpoints to
        checkpoint_every: (int|None) save a checkpoint every N steps
        checkpoint_every_minutes: (float|None) save a checkpoint every N minutes
        start_step: (int) the starting step to offset logging when resuming

    Returns:
        (model) our trained numpy model
        (list) a list containing our eval history, each entry is a dict with step, wall_time, loss, and scores
    """
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0
    eval_history = []
    last_checkpoint_time = time.time()
    try:
        for i, full_data in enumerate(prior):
            step = start_step + i
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            # if (torch.isnan(data[0]).any() or torch.isnan(data[1]).any()):
            #    continue
            data = (full_data["x"].to(device), full_data["y"][:, :train_test_split_index].to(device))
            targets = full_data["y"].to(device)

            output = model(data, train_test_split_index=train_test_split_index)
            targets = targets[:, train_test_split_index:]

            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, targets).mean()

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {step + 1}. Skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()
            total_loss = loss.cpu().detach().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            # evaluate
            if step % steps_per_eval == steps_per_eval - 1 and eval_func is not None:
                model.eval()
                optimizer.eval()

                classifier = NanoTabPFNClassifier(model, device)
                scores = eval_func(classifier)
                entry = {"step": step + 1, "wall_time": train_time, "loss": total_loss, **scores}
                eval_history.append(entry)
                score_str = " | ".join([f"{k} {v:7.4f}" for k, v in scores.items()])
                print(f"step {step + 1:5d} | time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")

                model.train()
                optimizer.train()
            elif step % steps_per_eval == steps_per_eval - 1 and eval_func is None:
                entry = {"step": step + 1, "wall_time": train_time, "loss": total_loss}
                eval_history.append(entry)
                print(f"step {step + 1:5d} | time {train_time:7.1f}s | loss {total_loss:7.4f}")

            # save checkpoint
            time_to_save = False
            if checkpoint_every_minutes is not None:
                current_wall_time = time.time()
                if (current_wall_time - last_checkpoint_time) / 60.0 >= checkpoint_every_minutes:
                    time_to_save = True
                    last_checkpoint_time = current_wall_time

            step_to_save = checkpoint_every is not None and (step + 1) % checkpoint_every == 0

            if checkpoint_dir and (time_to_save or step_to_save):
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{step + 1:05d}.pt")
                torch.save(model.state_dict(), checkpoint_path)

    except KeyboardInterrupt:
        pass

    # save final checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "final.pt")
        torch.save(model.state_dict(), final_path)

    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"[NanoTabPFN] Pretraining peak GPU memory allocated: {peak_mem_gb:.2f} GB")

    return model, eval_history


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename: str, num_steps: int, batch_size: int, device: torch.device | None = None):
        """Initialize loader."""
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device if device is not None else get_default_device()
        self.pointer = 0
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]  # pyright: ignore

    def __iter__(self):  # pyright: ignore
        """Yield batches."""
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                assert self.batch_size is not None
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()  # pyright: ignore
                num_datapoints_batch = f["num_datapoints"][self.pointer : end]  # pyright: ignore
                max_seq_in_batch = int(num_datapoints_batch.max())  # pyright: ignore
                x = torch.from_numpy(f["X"][self.pointer : end, :max_seq_in_batch, :num_features])  # pyright: ignore
                y = torch.from_numpy(f["y"][self.pointer : end, :max_seq_in_batch])  # pyright: ignore
                train_test_split_index = f["single_eval_pos"][self.pointer : end]  # pyright: ignore

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:  # pyright: ignore
                    print("Finished iteration over all stored datasets!")
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=train_test_split_index[0].item(),  # pyright: ignore
                )

    def __len__(self):
        """Return number of steps."""
        return self.num_steps


class NanopriorDataset(torch.utils.data.IterableDataset):
    """IterableDataset that generates synthetic prior data on the fly.

    Args:
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        max_seq_len (int): Maximum number of rows per dataset.
        max_features (int): Maximum number of features per dataset.
        max_classes (int): Maximum number of classes.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        max_seq_len: int = 3000,
        max_features: int = 45,
        max_classes: int = 10,
        device: torch.device | None = None,
    ):
        """Initialize dataset."""
        super().__init__()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_features = max_features
        self.max_classes = max_classes
        self.device = device if device is not None else get_default_device()

    def __iter__(self):
        """Yield batches."""
        import math

        from nanotabpfn.prior import rand_cat_sizes, rand_dataset_filtered

        worker_info = torch.utils.data.get_worker_info()
        # split steps across workers if in multi-process loading
        steps = self.num_steps if worker_info is None else math.ceil(self.num_steps / float(worker_info.num_workers))

        for _ in range(steps):
            n_samples = np.random.randint(100, self.max_seq_len + 1)
            n_features = np.random.randint(2, self.max_features + 1)
            n_classes = np.random.randint(2, self.max_classes + 1)

            x_cat_sizes = rand_cat_sizes(n_features)
            y_cat_sizes = [n_classes]

            xs, ys = [], []
            for _ in range(self.batch_size):
                tensors = rand_dataset_filtered(x_cat_sizes, y_cat_sizes, n_samples)
                x = torch.cat([tensors[f"x_{i}"] for i in range(len(x_cat_sizes))], dim=-1)
                y = tensors["y_0"].squeeze(-1)
                xs.append(x)
                ys.append(y)

            x_batch = torch.stack(xs, dim=0)
            y_batch = torch.stack(ys, dim=0)

            # 50% to 90% of samples used for training
            train_test_split_index = int(n_samples * np.random.uniform(0.5, 0.9))

            yield dict(
                x=x_batch.to(self.device),
                y=y_batch.to(self.device),
                train_test_split_index=train_test_split_index,
            )

    def __len__(self):
        """Return number of steps."""
        return self.num_steps


if __name__ == "__main__":
    set_randomness_seed(0)
    device = get_default_device()
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
    )
    dataset = NanopriorDataset(num_steps=2500, batch_size=32, device=device)
    prior = DataLoader(dataset, batch_size=None, num_workers=0)
    model, history = train(model, prior, lr=4e-3, steps_per_eval=25, eval_func=eval)
    print("Final evaluation:")
    print(eval(NanoTabPFNClassifier(model, device)))
