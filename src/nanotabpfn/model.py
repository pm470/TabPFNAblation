"""TabPFN Model definitions."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import LayerNorm, MultiheadAttention


class NanoTabPFNModel(nn.Module):
    """Core TabPFN Model."""

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        activation: str = "gelu",
    ):
        """Initializes the feature/target encoder, transformer stack and decoder."""
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size, activation=activation)
            )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs, activation=activation)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        """Forward pass for the model."""
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        # padds the y_train up to y by using the mean,
        # then converts scalar values to embeddings (B,R,1,E)
        y_src = self.target_encoder(y_src, num_rows)
        # concatenates the feature embeddings with the target embeddings
        # to give us the full table of embeddings (B,R,C,E))
        src_tensor = torch.cat([x_src, y_src], 2)
        # repeatedly applies the transformer block on (B,R,C,E)
        for block in self.transformer_blocks:
            src_tensor = block(src_tensor, train_test_split_index=train_test_split_index)
        # selects the target embeddings (B,num_targets,1,E)
        output = src_tensor[:, train_test_split_index:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    """Encodes features."""

    def __init__(self, embedding_size: int):
        """Creates the linear layer that we will use to embed our features."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """Normalizes all the features based on the mean and std of the features of the training data.

        Clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            train_test_split_index: (int) the number of datapoints in X_train
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        x = x.unsqueeze(-1).float()
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdim=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, unbiased=False, keepdim=True) + 1e-20
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    """Encodes targets."""

    def __init__(self, embedding_size: int):
        """Creates the linear layer that we will use to embed our targets."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """Padds up y_train to the full length of y using the mean per dataset and then embeds it using a linear layer.

        Args:
            y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
            num_rows: (int) the full length of y
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, 1, embedding_size), representing
                           the embeddings of the targets
        """
        # nan padding & nan handler instead?
        y_train = y_train.float()
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderLayer(nn.Module):
    """Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630."""

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        """Initialize TransformerEncoderLayer."""
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )

        self.mlp = create_mlp(
            embedding_size, mlp_hidden_size, embedding_size, activation=activation, device=device, dtype=dtype
        )

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """Takes the embeddings of the table as input and applies self-attention.

        Applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all
                                the embeddings for all the cells in the table
            train_test_split_index: (int) the length of X_train
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index], src[:, :train_test_split_index], src[:, :train_test_split_index]
        )[0]
        # test data attends to the training data
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:], src[:, :train_test_split_index], src[:, :train_test_split_index]
        )[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP after attention
        src = self.mlp(src) + src
        src = self.norm3(src)
        return src


def create_mlp(
    in_features: int, hidden_features: int, out_features: int, activation: str = "gelu", device=None, dtype=None
) -> nn.Module:
    """Factory function to create the appropriate MLP based on the activation type."""
    activation_name = activation.lower()
    if activation_name in ["swiglu", "geglu", "reglu"]:
        return GatedMLP(in_features, hidden_features, out_features, activation_name, device, dtype)
    return StandardMLP(in_features, hidden_features, out_features, activation_name, device, dtype)


class StandardMLP(nn.Module):
    """Standard 2-layer MLP."""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, activation: str, device=None, dtype=None
    ):
        """Initializes the standard MLP layers."""
        super().__init__()
        self.activation_name = activation
        self.linear1 = nn.Linear(in_features, hidden_features, device=device, dtype=dtype)
        self.linear2 = nn.Linear(hidden_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the MLP and standard activation function."""
        h = self.linear1(x)
        if self.activation_name == "gelu":
            h = F.gelu(h)
        elif self.activation_name == "relu":
            h = F.relu(h)
        elif self.activation_name in ["silu", "swish"]:
            h = F.silu(h)
        elif self.activation_name == "mish":
            h = F.mish(h)
        else:
            raise ValueError(f"Unsupported standard activation: {self.activation_name}")
        return self.linear2(h)


class GatedMLP(nn.Module):
    """Dynamically sizes hidden dimensions to maintain parameter count across activations."""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, activation: str, device=None, dtype=None
    ):
        """Initializes the gated MLP layers."""
        super().__init__()
        self.activation_name = activation

        # Baseline params (excluding out bias): in*H + H + H*out
        # Gated params (excluding out bias): 2*(in*H_new + H_new) + H_new*out
        target_params = hidden_features * (in_features + 1 + out_features)
        gated_divisor = 2 * in_features + 2 + out_features
        self.hidden_features = round(target_params / gated_divisor)

        self.linear_gate = nn.Linear(in_features, self.hidden_features, device=device, dtype=dtype)
        self.linear_up = nn.Linear(in_features, self.hidden_features, device=device, dtype=dtype)
        self.linear_down = nn.Linear(self.hidden_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the MLP and gated activation function."""
        gate = self.linear_gate(x)
        up = self.linear_up(x)
        if self.activation_name == "swiglu":
            gate = F.silu(gate)
        elif self.activation_name == "geglu":
            gate = F.gelu(gate)
        elif self.activation_name == "reglu":
            gate = F.relu(gate)
        else:
            raise ValueError(f"Unsupported gated activation: {self.activation_name}")
        return self.linear_down(gate * up)


class Decoder(nn.Module):
    """Decodes embeddings into logits."""

    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int, activation: str = "gelu"):
        """Initializes the linear layers for use in the forward."""
        super().__init__()
        self.mlp = create_mlp(embedding_size, mlp_hidden_size, num_outputs, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies an MLP to the embeddings to get the logits.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embedding_size)

        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_outputs)
        """
        return self.mlp(x)


class NanoTabPFNClassifier:
    """scikit-learn like interface."""

    def __init__(
        self,
        model: NanoTabPFNModel,
        device: torch.device,
        max_train_samples: int | None = None,
        max_total_samples: int | None = None,
        n_ensemble: int = 1,
    ):
        """Initialize classifier."""
        self.model = model.to(device)
        self.device = device
        self.max_train_samples = max_train_samples
        self.max_total_samples = max_total_samples
        self.n_ensemble = n_ensemble

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Stores X_train and y_train for later use, also computes the highest class number occuring in num_classes."""
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Creates (x,y), runs it through our PyTorch Model.

        Subsamples training data and chunks test data to avoid OOM.
        Cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities.
        """
        # Determine limits based on feature count to avoid quadratic memory OOM in attention
        if self.max_train_samples is not None and self.max_total_samples is not None:
            max_train_samples = self.max_train_samples
            max_total_samples = self.max_total_samples
        else:
            num_features = self.X_train.shape[1]
            col_size = num_features + 1
            num_heads = 4
            if len(self.model.transformer_blocks) > 0:
                num_heads = self.model.transformer_blocks[0].self_attention_between_datapoints.num_heads  # pyright: ignore[reportAttributeAccessIssue]

            # Target peak memory of 30.0 GiB for the attention weights tensor.
            # The actual peak is ~2x this (scores + softmax output coexist briefly),
            # so this targets ~60 GiB peak — perfect for 80+ GB GPUs like the A100.
            # Memory per attention = col_size * num_heads * (seq_len ** 2) * 4 bytes
            max_attn_bytes = 30.0 * (1024**3)
            max_seq_len = int(np.sqrt(max_attn_bytes / (col_size * num_heads * 4)))

            # Clip max_seq_len to a reasonable range [1000, 10000]
            max_seq_len = max(1000, min(10000, max_seq_len))

            max_train_samples = max_seq_len
            max_total_samples = int(max_seq_len * 1.25)

        # 1. Chunk test data
        max_test_chunk = max_total_samples - min(len(self.X_train), max_train_samples)
        # Fallback in case max_test_chunk is very small or negative
        max_test_chunk = max(100, max_test_chunk)

        all_ensemble_probs = []
        # If the dataset is small enough, no subsampling is needed.
        # Since nanoTabPFN does not yet do feature/label permutations, running
        # multiple identical ensembles would be a waste of compute.
        # Note: We keep this ensembling logic because while pretraining only targets up to 3000 rows
        # (which easily fits in context), the full TabArena benchmark evaluates on datasets with 10k+ rows,
        # making ensembling necessary to utilize the full training folds without OOMing.
        actual_ensemble_size = self.n_ensemble if len(self.X_train) > max_train_samples else 1

        for ensemble_idx in range(actual_ensemble_size):
            if len(self.X_train) > max_train_samples:
                # 2. Stratified subsample training context if it's too large
                try:
                    X_train_sub, _, y_train_sub, _ = train_test_split(
                        self.X_train,
                        self.y_train,
                        train_size=max_train_samples,
                        stratify=self.y_train,
                        random_state=42 + ensemble_idx,
                    )
                except ValueError:
                    # Fallback to purely random subset if a class has too few samples to stratify
                    rng = np.random.default_rng(42 + ensemble_idx)
                    indices = rng.choice(len(self.X_train), max_train_samples, replace=False)
                    X_train_sub = self.X_train[indices]
                    y_train_sub = self.y_train[indices]
            else:
                X_train_sub = self.X_train
                y_train_sub = self.y_train

            all_probs = []
            for i in range(0, len(X_test), max_test_chunk):
                X_test_chunk = X_test[i : i + max_test_chunk]
                x = np.concatenate((X_train_sub, X_test_chunk))
                y = y_train_sub

                # Use autocast for mixed precision (saves memory, allows larger context)
                # bfloat16 has the same memory footprint as float16 but shares float32's
                # exponent range (max ~3.4e38), avoiding the overflow→NaN issue of float16
                device_type = self.device.type if self.device.type != "mps" else "cpu"
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device_type, dtype=torch.bfloat16, enabled=self.device.type != "cpu"
                    ):
                        x_tensor = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
                        y_tensor = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
                        out = self.model((x_tensor, y_tensor), train_test_split_index=len(X_train_sub)).squeeze(0)

                    # Compute softmax in float32 outside autocast for numerical stability
                    out = out[:, : self.num_classes].to(torch.float32)
                    probs = F.softmax(out, dim=1).cpu().numpy()
                    all_probs.append(probs)

            all_ensemble_probs.append(np.concatenate(all_probs, axis=0))

        # Average probabilities across all ensemble members
        return np.mean(all_ensemble_probs, axis=0)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)
