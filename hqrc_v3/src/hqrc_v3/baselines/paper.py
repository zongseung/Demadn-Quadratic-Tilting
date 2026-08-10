"""Factories for the five frozen manuscript baselines."""

from __future__ import annotations

from hqrc_v3.baselines.classical import ClassicalBaseline, make_classical_baseline
from hqrc_v3.baselines.config import (
    MODEL_NAMES,
    NeuralPaperConfig,
    PaperBaselineConfig,
    PaperBaselineConfigError,
)
from hqrc_v3.baselines.protocol import BaselineFactory
from hqrc_v3.baselines.sequence import SequenceTrainingConfig, TorchBaselineFactory


def _sequence_config(config: NeuralPaperConfig) -> SequenceTrainingConfig:
    return SequenceTrainingConfig(
        hidden_size=config.hidden_size,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        seeds=config.seeds,
    )


def make_paper_factory(
    config: PaperBaselineConfig,
    model: str,
) -> BaselineFactory:
    """Create exactly one immutable paper factory without search or substitution."""

    if not isinstance(config, PaperBaselineConfig):
        raise TypeError("config must be a PaperBaselineConfig")
    if model not in MODEL_NAMES or model not in config.models:
        raise PaperBaselineConfigError(f"unknown paper model {model!r}")
    if model == "xgboost":
        return make_classical_baseline(model, config.xgboost.to_estimator_params())
    if model == "lightgbm":
        return make_classical_baseline(
            model,
            config.lightgbm.to_estimator_params(),
            early_stopping_rounds=config.lightgbm.early_stopping_rounds,
        )
    if model == "svr":
        return make_classical_baseline(model, config.svr.to_estimator_params())
    neural = config.seq2seq_lstm if model == "seq2seq_lstm" else config.transformer
    return TorchBaselineFactory(model, _sequence_config(neural))


__all__ = ["ClassicalBaseline", "make_paper_factory"]
