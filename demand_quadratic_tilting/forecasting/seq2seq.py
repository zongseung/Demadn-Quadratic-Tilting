"""Shared encoder-decoder implementation for LSTM and GRU baselines."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


RNNKind = Literal["lstm", "gru"]
DecoderMode = Literal["autoregressive", "context"]


@dataclass(frozen=True)
class Seq2SeqTrainingConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 10
    teacher_forcing_ratio: float = 0.5
    gradient_clip: float = 1.0
    min_delta: float = 1e-5
    decoder_mode: DecoderMode = "autoregressive"


class Seq2SeqRNN(nn.Module):
    """Autoregressive Seq2Seq forecaster with future-known covariates."""

    def __init__(
        self,
        kind: RNNKind,
        encoder_input_size: int,
        known_covariate_size: int,
        horizon_hours: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        decoder_mode: DecoderMode = "autoregressive",
    ) -> None:
        super().__init__()
        if kind not in ("lstm", "gru"):
            raise ValueError("kind must be 'lstm' or 'gru'")
        if decoder_mode not in ("autoregressive", "context"):
            raise ValueError("decoder_mode must be 'autoregressive' or 'context'")
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.kind = kind
        self.decoder_mode = decoder_mode
        self.horizon_hours = horizon_hours
        self.encoder = recurrent(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.decoder = recurrent(
            input_size=1 + known_covariate_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        if kind == "lstm":
            self._initialize_lstm_forget_gates()

    def _initialize_lstm_forget_gates(self) -> None:
        """Start LSTM forget gates open enough for a 168-step encoder."""

        for recurrent in (self.encoder, self.decoder):
            for layer in range(recurrent.num_layers):
                bias_ih = getattr(recurrent, f"bias_ih_l{layer}")
                bias_hh = getattr(recurrent, f"bias_hh_l{layer}")
                hidden_size = recurrent.hidden_size
                with torch.no_grad():
                    bias_ih[hidden_size : 2 * hidden_size].fill_(1.0)
                    bias_hh[hidden_size : 2 * hidden_size].zero_()

    def forward(
        self,
        history: torch.Tensor,
        future_known: torch.Tensor,
        teacher_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if future_known.shape[1] != self.horizon_hours:
            raise ValueError("future-known sequence has the wrong horizon")
        _, state = self.encoder(history)
        if self.decoder_mode == "context":
            last_observation = history[:, -1:, 0:1].expand(-1, self.horizon_hours, -1)
            decoder_input = torch.cat((last_observation, future_known), dim=2)
            decoded, _ = self.decoder(decoder_input, state)
            return self.output_layer(decoded).squeeze(-1)

        previous = history[:, -1, 0:1]
        predictions: list[torch.Tensor] = []

        for step in range(self.horizon_hours):
            decoder_input = torch.cat(
                (previous, future_known[:, step, :]), dim=1
            ).unsqueeze(1)
            decoded, state = self.decoder(decoder_input, state)
            prediction = self.output_layer(decoded[:, -1, :])
            predictions.append(prediction)

            if teacher_targets is not None and teacher_forcing_ratio > 0:
                use_teacher = (
                    torch.rand(history.shape[0], 1, device=history.device)
                    < teacher_forcing_ratio
                )
                previous = torch.where(
                    use_teacher,
                    teacher_targets[:, step : step + 1],
                    prediction,
                )
            else:
                previous = prediction
        return torch.cat(predictions, dim=1)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_seq2seq_device(kind: RNNKind, device: str = "auto") -> torch.device:
    """Resolve a reproducible device for a recurrent baseline.

    PyTorch 2.8 on Apple MPS can update an internal packed LSTM weight buffer
    without producing a state dict that reproduces those predictions in a new
    process.  GRU checkpoints do not show this issue.  Keep LSTM on CPU on
    Apple Silicon until the backend round-trip is reliable.
    """

    resolved = resolve_device(device)
    if kind == "lstm" and resolved.type == "mps":
        return torch.device("cpu")
    return resolved


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    history: np.ndarray,
    future_known: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(history).float(),
        torch.from_numpy(future_known).float(),
        torch.from_numpy(target).float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def _mean_loss(
    model: Seq2SeqRNN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for history, future_known, target in loader:
        history = history.to(device)
        future_known = future_known.to(device)
        target = target.to(device)
        prediction = model(history, future_known)
        squared_error += torch.sum((prediction - target) ** 2).item()
        count += target.numel()
    return squared_error / max(count, 1)


def fit_seq2seq(
    kind: RNNKind,
    train_history: np.ndarray,
    train_future_known: np.ndarray,
    train_target: np.ndarray,
    validation_history: np.ndarray,
    validation_future_known: np.ndarray,
    validation_target: np.ndarray,
    *,
    training: Seq2SeqTrainingConfig | None = None,
    random_seed: int = 2025,
    device: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> tuple[Seq2SeqRNN, list[dict[str, float]]]:
    """Fit a Seq2Seq baseline with validation early stopping."""

    training = training or Seq2SeqTrainingConfig()
    seed_everything(random_seed)
    torch_device = resolve_seq2seq_device(kind, device)
    model = Seq2SeqRNN(
        kind=kind,
        encoder_input_size=train_history.shape[2],
        known_covariate_size=train_future_known.shape[2],
        horizon_hours=train_target.shape[1],
        hidden_size=training.hidden_size,
        num_layers=training.num_layers,
        dropout=training.dropout,
        decoder_mode=training.decoder_mode,
    ).to(torch_device)
    optimizer = AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, training.patience // 3),
    )
    train_loader = _loader(
        train_history,
        train_future_known,
        train_target,
        training.batch_size,
        True,
        random_seed,
    )
    validation_loader = _loader(
        validation_history,
        validation_future_known,
        validation_target,
        training.batch_size,
        False,
        random_seed,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    records: list[dict[str, float]] = []
    for epoch in range(1, training.max_epochs + 1):
        model.train()
        total_squared_error = 0.0
        total_count = 0
        for history, future_known, target in train_loader:
            history = history.to(torch_device)
            future_known = future_known.to(torch_device)
            target = target.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                history,
                future_known,
                teacher_targets=target,
                teacher_forcing_ratio=training.teacher_forcing_ratio,
            )
            loss = torch.mean((prediction - target) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
            optimizer.step()
            total_squared_error += torch.sum((prediction - target) ** 2).item()
            total_count += target.numel()

        train_loss = total_squared_error / max(total_count, 1)
        validation_loss = _mean_loss(model, validation_loader, torch_device)
        scheduler.step(validation_loss)
        records.append(
            {
                "epoch": float(epoch),
                "train_mse_scaled": float(train_loss),
                "validation_mse_scaled": float(validation_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if progress is not None and (epoch == 1 or epoch % 5 == 0):
            progress(
                f"epoch={epoch} train_mse={train_loss:.5f} "
                f"validation_mse={validation_loss:.5f}"
            )

        if validation_loss < best_loss - training.min_delta:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in deepcopy(model.state_dict()).items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= training.patience:
            if progress is not None:
                progress(
                    f"early_stop epoch={epoch} best_validation_mse={best_loss:.5f}"
                )
            break

    if best_state is None:
        raise RuntimeError("Seq2Seq training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(torch_device)
    model.eval()
    return model, records


@torch.no_grad()
def predict_seq2seq(
    model: Seq2SeqRNN,
    history: np.ndarray,
    future_known: np.ndarray,
    *,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    torch_device = resolve_seq2seq_device(model.kind, device)
    model.to(torch_device)
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(history).float(),
        torch.from_numpy(future_known).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    for history_batch, future_batch in loader:
        prediction = model(
            history_batch.to(torch_device),
            future_batch.to(torch_device),
        )
        predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def checkpoint_payload(
    model: Seq2SeqRNN,
    training: Seq2SeqTrainingConfig,
    records: list[dict[str, float]],
) -> dict[str, object]:
    return {
        "kind": model.kind,
        "decoder_mode": model.decoder_mode,
        "horizon_hours": model.horizon_hours,
        "training_config": asdict(training),
        "training_history": records,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
    }
