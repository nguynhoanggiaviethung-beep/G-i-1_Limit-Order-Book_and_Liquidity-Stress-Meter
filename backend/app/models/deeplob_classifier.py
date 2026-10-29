"""Optional sequence classifier using a spatial LOB CNN and temporal LSTM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEEPLOB_INPUT_COLUMNS = tuple(
    [f"{side}_price_{level}" for side in ("bid", "ask") for level in range(1, 11)]
    + [f"{side}_qty_{level}" for side in ("bid", "ask") for level in range(1, 11)]
)


def build_lob_sequences(
    frame: pd.DataFrame,
    endpoints: np.ndarray | list[int],
    lookback: int,
    columns: tuple[str, ...] = DEEPLOB_INPUT_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build finite, contiguous one-second LOB windows ending at each origin."""
    if lookback < 2:
        raise ValueError("DeepLOB lookback must be at least two seconds")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"DeepLOB input is missing book columns: {', '.join(missing[:4])}")
    endpoints = np.asarray(endpoints, dtype=int)
    endpoints = endpoints[(endpoints - lookback + 1 >= 0) & (endpoints < len(frame))]
    if not len(endpoints):
        return np.empty((0, lookback, len(columns)), dtype=np.float32), np.asarray([], dtype=int)
    # Convert only the rows inside some window, not the whole (possibly months-long) frame.
    rows, positions = np.unique(endpoints[:, None] + np.arange(-lookback + 1, 1), return_inverse=True)
    positions = positions.reshape(len(endpoints), lookback)
    subset = frame.iloc[rows]
    values = subset.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    times = pd.to_datetime(subset["timestamp_utc"], utc=True, errors="coerce").astype("int64").to_numpy()
    valid = subset.get("book_valid", pd.Series(True, index=subset.index)).fillna(False).astype(bool).to_numpy()
    selected_values, selected_endpoints = [], []
    second_ns = 1_000_000_000
    for endpoint, segment in zip(endpoints, positions):
        if not valid[segment].all() or not np.all(np.diff(times[segment]) == second_ns):
            continue
        window = values[segment]
        if window.shape != (lookback, len(columns)) or not np.isfinite(window).all():
            continue
        if tuple(columns) == DEEPLOB_INPUT_COLUMNS:
            window = window.copy()
            mid = (window[:, 0] + window[:, 10]) / 2.0
            if np.any(mid <= 0):
                continue
            # Remove nominal-price scale while retaining the shape of each side.
            window[:, :20] = (window[:, :20] / mid[:, None] - 1.0) * 10000.0
        selected_values.append(window)
        selected_endpoints.append(endpoint)
    if not selected_values:
        return np.empty((0, lookback, len(columns)), dtype=np.float32), np.asarray([], dtype=int)
    return np.stack(selected_values).astype(np.float32, copy=False), np.asarray(selected_endpoints, dtype=int)


class DeepLOBClassifier:
    """Small deterministic spatial-CNN/LSTM multiclass or binary classifier.

    Inputs are ``(samples, lookback, 40)``: ten price and ten quantity
    levels on each side. PyTorch is an optional runtime dependency.
    """

    def __init__(
        self,
        lookback: int = 100,
        random_state: int = 20261021,
        hidden_size: int = 32,
        batch_size: int = 128,
        max_epochs: int = 12,
        patience: int = 3,
        learning_rate: float = 0.001,
    ) -> None:
        if lookback < 2 or hidden_size < 8 or batch_size < 1 or max_epochs < 1 or patience < 1:
            raise ValueError("DeepLOB model dimensions and training limits must be positive")
        self.lookback = int(lookback)
        self.random_state = int(random_state)
        self.hidden_size = int(hidden_size)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.learning_rate = float(learning_rate)
        self.scaler = StandardScaler()
        self.classes_: np.ndarray | None = None
        self.model = None
        self.n_features_in_: int | None = None
        self.parameter_count_: int | None = None

    @staticmethod
    def _torch():
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("Install the optional PyTorch dependency from requirements-dl.txt to run DeepLOB") from exc
        return torch, nn, DataLoader, TensorDataset

    def _make_network(self, nn, input_features: int, classes: int):
        import torch

        hidden = self.hidden_size

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.book_cnn = nn.Sequential(
                    nn.Conv2d(4, 16, kernel_size=(1, 2)),
                    nn.BatchNorm2d(16),
                    nn.GELU(),
                    nn.Conv2d(16, 16, kernel_size=(1, 2)),
                    nn.GELU(),
                    nn.Conv2d(16, hidden, kernel_size=(3, 1), padding=(1, 0)),
                    nn.GELU(),
                )
                # Four channels are bid/ask prices and quantities across ten levels.
                self.recurrent = nn.LSTM(hidden * 8, hidden, batch_first=True)
                self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, classes))

            def forward(self, values):
                # [batch,time,40] -> [batch,4,time,10]; CNN learns local level patterns.
                lob = values.reshape(values.shape[0], values.shape[1], 4, 10).permute(0, 2, 1, 3)
                encoded = self.book_cnn(lob).permute(0, 2, 1, 3).flatten(start_dim=2)
                recurrent, _ = self.recurrent(encoded)
                return self.head(recurrent[:, -1, :])

        return Network()

    def _scale(self, values: np.ndarray) -> np.ndarray:
        shape = values.shape
        return self.scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    def fit(self, x, y):
        torch, nn, DataLoader, TensorDataset = self._torch()
        values = np.asarray(x, dtype=np.float32)
        target = np.asarray(y)
        if values.ndim != 3 or values.shape[1] != self.lookback:
            raise ValueError(f"x must have shape (samples, {self.lookback}, input_features)")
        if values.shape[2] != 40:
            raise ValueError("DeepLOB expects 40 channels: ten price and ten quantity levels on each side")
        if len(values) != len(target) or len(target) < 60 or not np.isfinite(values).all():
            raise ValueError("DeepLOB requires at least 60 aligned samples with finite features")
        self.classes_ = np.asarray(sorted(set(target.tolist()), key=str), dtype=object)
        if len(self.classes_) < 2:
            raise ValueError("DeepLOB training requires at least two classes")
        mapping = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([mapping[label] for label in target], dtype=np.int64)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.n_features_in_ = int(values.shape[2])
        validation_size = max(1, int(round(len(values) * 0.1)))
        split = len(values) - validation_size
        self.scaler.fit(values[:split].reshape(-1, self.n_features_in_))
        scaled = self._scale(values)
        train_x = torch.from_numpy(scaled[:split])
        train_y = torch.from_numpy(encoded[:split])
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=min(self.batch_size, len(train_y)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        valid_x = torch.from_numpy(scaled[split:])
        valid_y = torch.from_numpy(encoded[split:])
        self.model = self._make_network(nn, self.n_features_in_, len(self.classes_))
        self.parameter_count_ = int(sum(parameter.numel() for parameter in self.model.parameters()))
        counts = np.bincount(encoded[:split], minlength=len(self.classes_)).astype(np.float32)
        weights = len(train_y) / np.maximum(counts, 1) / len(self.classes_)
        criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        best_loss, best_weights, stale_epochs = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                score = float(criterion(self.model(valid_x), valid_y).item())
            if score < best_loss - 1e-5:
                best_loss = score
                best_weights = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        return self

    def predict_proba(self, x):
        if self.model is None or self.classes_ is None or self.n_features_in_ is None:
            raise RuntimeError("fit must be called before predict_proba")
        torch, _, _, _ = self._torch()
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (self.lookback, self.n_features_in_) or not np.isfinite(values).all():
            raise ValueError(f"x must have finite shape (samples, {self.lookback}, {self.n_features_in_})")
        scaled = self._scale(values)
        self.model.eval()
        output = []
        with torch.no_grad():
            for start in range(0, len(scaled), 1024):
                logits = self.model(torch.from_numpy(scaled[start : start + 1024]))
                output.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(output) if output else np.empty((0, len(self.classes_)), dtype=float)

    def predict(self, x):
        probabilities = self.predict_proba(x)
        return np.asarray(self.classes_[np.argmax(probabilities, axis=1)].tolist())

    def save_state(self, path: str) -> None:
        """Save portable model weights and the training-only feature scaler."""
        if self.model is None or self.classes_ is None:
            raise RuntimeError("fit must be called before save_state")
        import torch

        torch.save({
            "state_dict": self.model.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "classes": self.classes_.tolist(),
            "lookback": self.lookback,
            "input_features": self.n_features_in_,
            "parameter_count": self.parameter_count_,
        }, path)
