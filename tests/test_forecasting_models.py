import numpy as np
import torch

from demand_quadratic_tilting.forecasting.ml import fit_direct_multi_horizon
from demand_quadratic_tilting.forecasting.seq2seq import (
    Seq2SeqRNN,
    resolve_seq2seq_device,
)


def test_direct_svr_supports_arbitrary_multi_horizon_shape() -> None:
    rng = np.random.default_rng(2025)
    x_train = rng.normal(size=(30, 8)).astype(np.float32)
    y_train = rng.normal(size=(30, 3)).astype(np.float32)
    x_validation = rng.normal(size=(10, 8)).astype(np.float32)
    y_validation = rng.normal(size=(10, 3)).astype(np.float32)

    model = fit_direct_multi_horizon(
        "svr",
        x_train,
        y_train,
        x_validation,
        y_validation,
        n_jobs=1,
        quick=True,
    )
    assert model.predict(x_validation).shape == (10, 3)


def test_random_forest_uses_native_multi_output() -> None:
    rng = np.random.default_rng(2025)
    x_train = rng.normal(size=(30, 8)).astype(np.float32)
    y_train = rng.normal(size=(30, 3)).astype(np.float32)
    x_validation = rng.normal(size=(10, 8)).astype(np.float32)
    y_validation = rng.normal(size=(10, 3)).astype(np.float32)

    model = fit_direct_multi_horizon(
        "random_forest",
        x_train,
        y_train,
        x_validation,
        y_validation,
        n_jobs=1,
        quick=True,
    )
    assert model.horizon_hours == 3
    assert len(model.estimators) == 1
    assert model.predict(x_validation).shape == (10, 3)


def test_lstm_and_gru_seq2seq_shapes() -> None:
    history = torch.randn(4, 12, 7)
    future_known = torch.randn(4, 5, 4)
    targets = torch.randn(4, 5)

    for kind, decoder_mode in (
        ("lstm", "autoregressive"),
        ("lstm", "context"),
        ("gru", "autoregressive"),
        ("gru", "context"),
    ):
        model = Seq2SeqRNN(
            kind=kind,
            encoder_input_size=7,
            known_covariate_size=4,
            horizon_hours=5,
            hidden_size=8,
            num_layers=1,
            decoder_mode=decoder_mode,
        )
        inference = model(history, future_known)
        training = model(
            history,
            future_known,
            teacher_targets=targets,
            teacher_forcing_ratio=0.5,
        )
        assert inference.shape == (4, 5)
        assert training.shape == (4, 5)


def test_mps_lstm_uses_reproducible_cpu_fallback() -> None:
    assert resolve_seq2seq_device("lstm", "mps").type == "cpu"
    assert resolve_seq2seq_device("gru", "mps").type == "mps"
