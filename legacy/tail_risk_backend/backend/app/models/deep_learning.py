"""Optional sequence regressors for point-in-time realized-volatility forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


DEEP_LEARNING_MODELS = ("lstm", "gru", "transformer")


def build_model(
    model_name: str,
    input_size: int,
    hidden_size: int = 32,
    num_layers: int = 2,
    dropout: float = 0.1,
    n_heads: int = 4,
    max_sequence_length: int = 30,
):
    """Build an LSTM, GRU or causal Transformer encoder in PyTorch."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install requirements-dl.txt to use deep-learning models") from exc

    if model_name not in DEEP_LEARNING_MODELS:
        raise ValueError(f"Unsupported deep-learning model: {model_name}")
    if input_size < 1 or hidden_size < 4 or num_layers < 1 or max_sequence_length < 1:
        raise ValueError("model dimensions and sequence length must be positive")

    class SequenceNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = model_name
            if model_name in {"lstm", "gru"}:
                layer = nn.LSTM if model_name == "lstm" else nn.GRU
                self.recurrent = layer(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.encoder = None
                self.input_projection = None
                self.position = None
            else:
                if hidden_size % n_heads:
                    raise ValueError("hidden_size must be divisible by n_heads for transformer")
                self.recurrent = None
                self.input_projection = nn.Linear(input_size, hidden_size)
                self.position = nn.Parameter(torch.zeros(1, max_sequence_length, hidden_size))
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=n_heads,
                    dim_feedforward=hidden_size * 2,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, max(hidden_size // 2, 4)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(hidden_size // 2, 4), 1),
            )

        def forward(self, values):
            if self.name in {"lstm", "gru"}:
                encoded, _ = self.recurrent(values)
            else:
                length = values.shape[1]
                if length > self.position.shape[1]:
                    raise ValueError("input sequence exceeds configured maximum length")
                projected = self.input_projection(values) + self.position[:, :length]
                # A causal mask ensures each timestep only attends to its past.
                mask = torch.triu(
                    torch.ones(length, length, device=values.device, dtype=torch.bool),
                    diagonal=1,
                )
                encoded = self.encoder(projected, mask=mask)
            return self.head(encoded[:, -1, :]).squeeze(-1)

    return SequenceNetwork()


class DeepSequenceRegressor:
    """Small deterministic PyTorch regressor using chronological validation."""

    def __init__(
        self,
        model_name: str,
        random_state: int = 20260930,
        lookback: int = 30,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        max_epochs: int = 60,
        patience: int = 8,
    ):
        if model_name not in DEEP_LEARNING_MODELS:
            raise ValueError(f"Unsupported deep-learning model: {model_name}")
        if lookback < 2 or batch_size < 1 or max_epochs < 1 or patience < 1:
            raise ValueError("lookback, batch_size, max_epochs and patience must be positive")
        self.model_name = model_name
        self.random_state = int(random_state)
        self.lookback = int(lookback)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.scaler = StandardScaler()
        self.model = None
        self.n_features_in_ = None

    def fit(self, x, y):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("Install requirements-dl.txt to use deep-learning models") from exc

        values = np.asarray(x, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32).reshape(-1)
        if values.ndim != 3 or values.shape[1] != self.lookback:
            raise ValueError(f"x must have shape (samples, {self.lookback}, features)")
        if len(values) != len(target) or len(target) < 30:
            raise ValueError("deep-learning fit requires at least 30 aligned samples")
        if not np.isfinite(values).all() or not np.isfinite(target).all() or np.any(target <= 0):
            raise ValueError("features and positive volatility targets must be finite")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.n_features_in_ = int(values.shape[2])
        validation_size = max(1, int(round(len(target) * 0.1)))
        if len(target) - validation_size < 20:
            validation_size = 0
        split = len(target) - validation_size
        fit_flat = values[:split].reshape(-1, self.n_features_in_)
        self.scaler.fit(fit_flat)
        scaled = self.scaler.transform(values.reshape(-1, self.n_features_in_)).reshape(values.shape).astype(np.float32)
        log_target = np.log(target).astype(np.float32)

        train_x = torch.from_numpy(scaled[:split])
        train_y = torch.from_numpy(log_target[:split])
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=min(self.batch_size, len(train_y)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        valid_x = torch.from_numpy(scaled[split:]) if validation_size else None
        valid_y = torch.from_numpy(log_target[split:]) if validation_size else None

        self.model = build_model(
            self.model_name,
            self.n_features_in_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_sequence_length=self.lookback,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()
        best_loss = float("inf")
        best_weights = None
        stale_epochs = 0
        for _ in range(self.max_epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                if validation_size:
                    score = float(criterion(self.model(valid_x), valid_y).item())
                else:
                    score = float(criterion(self.model(train_x), train_y).item())
            if score < best_loss - 1e-6:
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

    def predict(self, x):
        if self.model is None or self.n_features_in_ is None:
            raise RuntimeError("fit must be called before predict")
        import torch

        values = np.asarray(x, dtype=np.float32)
        if values.ndim == 2:
            values = values.reshape(1, *values.shape)
        if values.ndim != 3 or values.shape[1] != self.lookback or values.shape[2] != self.n_features_in_:
            raise ValueError(
                f"x must have shape (samples, {self.lookback}, {self.n_features_in_})"
            )
        if not np.isfinite(values).all():
            raise ValueError("prediction features must be finite")
        shape = values.shape
        scaled = self.scaler.transform(values.reshape(-1, self.n_features_in_)).reshape(shape)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.from_numpy(scaled.astype(np.float32))).cpu().numpy()
        return np.exp(np.clip(predictions, -30, 30))
