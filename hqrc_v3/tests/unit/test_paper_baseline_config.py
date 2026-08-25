"""Frozen manuscript-baseline configuration contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import hqrc_v3.baselines as baselines
import hqrc_v3.baselines.classical as classical
from hqrc_v3.baselines.classical import ClassicalBaseline
from hqrc_v3.baselines.config import (
    B1W_WINDOW_COLUMNS,
    B1W_WINDOW_VERSION,
    MODEL_NAMES,
    PAPER_SEEDS,
    PaperBaselineConfig,
    PaperBaselineConfigError,
    load_paper_baselines,
)
from hqrc_v3.baselines.paper import (
    make_paper_factory,
    run_paper_final_stage,
    run_paper_oof_stage,
)
from hqrc_v3.baselines.sequence import (
    Seq2SeqLSTM,
    TimeSeriesTransformer,
    TorchBaselineFactory,
)
from hqrc_v3.contracts import ForecastMatrix
from hqrc_v3.oof import chronological_validation_tail
from hqrc_v3.provenance import ArtifactMismatch, file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs/model_spaces.toml"


def _daily_matrix(start: str, days: int) -> ForecastMatrix:
    origins = np.arange(
        np.datetime64(start),
        np.datetime64(start) + np.timedelta64(days, "D"),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    target_times = origins[:, None] + np.arange(24).astype("timedelta64[h]")
    return ForecastMatrix(
        origins=origins,
        target_times=target_times,
        history=np.zeros((days, 168, 1), dtype=float),
        future=np.zeros((days, 24, 1), dtype=float),
        target=np.zeros((days, 24), dtype=float),
        history_columns=("load_mw",),
        future_columns=("hour",),
    )


def test_versioned_config_loads_only_the_five_exact_paper_models() -> None:
    config = load_paper_baselines(
        MODEL_CONFIG,
        expected_sha256=file_sha256(MODEL_CONFIG),
    )

    assert config.models == (
        "xgboost",
        "lightgbm",
        "svr",
        "seq2seq_lstm",
        "transformer",
    )
    assert config.models == MODEL_NAMES
    assert config.validation_days == 61
    assert config.schema_version == 2
    assert config.preprocessing.version == "causal-v1"
    assert config.preprocessing.future_path_hours == 24
    assert len(config.preprocessing.b0_future_columns) == 7
    assert len(config.preprocessing.b1_only_columns) == 7
    assert config.seq2seq_lstm.seeds == config.transformer.seeds == PAPER_SEEDS


def test_reviewer_b1w_schema_is_frozen_outside_the_toml_contract() -> None:
    config = load_paper_baselines(MODEL_CONFIG)

    assert config.preprocessing.b1_only_columns == (
        "is_public_holiday",
        "official_sequence_position",
        "seollal_distance",
        "chuseok_distance",
        "is_substitute_or_temporary_holiday",
        "is_seollal",
        "is_chuseok",
    )
    assert B1W_WINDOW_COLUMNS == ("is_seollal_window", "is_chuseok_window")
    assert B1W_WINDOW_VERSION == "official-sequence-buffer-v1"


def test_paper_baseline_api_is_available_from_the_package_namespace() -> None:
    assert baselines.PaperBaselineConfig is PaperBaselineConfig
    assert baselines.load_paper_baselines is load_paper_baselines
    assert baselines.make_paper_factory is make_paper_factory
    assert baselines.run_paper_oof_stage is run_paper_oof_stage
    assert baselines.run_paper_final_stage is run_paper_final_stage


def test_versioned_config_freezes_every_manuscript_parameter() -> None:
    config = load_paper_baselines(MODEL_CONFIG)

    assert config.xgboost.to_estimator_params() == {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "early_stopping_rounds": 35,
    }
    assert config.lightgbm.to_estimator_params() == {
        "objective": "regression",
        "n_estimators": 800,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }
    assert config.lightgbm.early_stopping_rounds == 35
    assert config.svr.to_estimator_params() == {
        "kernel": "rbf",
        "C": 10.0,
        "epsilon": 0.05,
        "gamma": "scale",
        "cache_size": 512.0,
    }
    for neural in (config.seq2seq_lstm, config.transformer):
        assert neural.hidden_size == 64
        assert neural.dropout == 0.1
        assert neural.learning_rate == 1e-3
        assert neural.batch_size == 64
        assert neural.epochs == 60
        assert neural.patience == 8
    assert config.seq2seq_lstm.layers == 2
    assert config.transformer.layers == 1
    assert config.transformer.heads == 4


def test_frozen_config_hash_mismatch_fails_closed() -> None:
    with pytest.raises(ArtifactMismatch, match="model config SHA-256"):
        load_paper_baselines(MODEL_CONFIG, expected_sha256="0" * 64)


def test_loader_rejects_a_rehashed_changed_seed_set(tmp_path: Path) -> None:
    changed = tmp_path / "changed.toml"
    changed.write_text(
        MODEL_CONFIG.read_text(encoding="utf-8").replace(
            "seeds = [11, 23, 37, 41, 53]",
            "seeds = [11, 23, 37, 41, 59]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(PaperBaselineConfigError, match="seeds"):
        load_paper_baselines(changed, expected_sha256=file_sha256(changed))


def test_exact_config_selects_five_immutable_factories() -> None:
    config = load_paper_baselines(MODEL_CONFIG)

    factories = tuple(make_paper_factory(config, model) for model in config.models)

    assert tuple(factory.name for factory in factories) == config.models
    assert all(isinstance(factory, ClassicalBaseline) for factory in factories[:3])
    assert all(isinstance(factory, TorchBaselineFactory) for factory in factories[3:])
    assert factories[0].params == config.xgboost.to_estimator_params()
    assert factories[1].params == config.lightgbm.to_estimator_params()
    assert factories[2].params == config.svr.to_estimator_params()
    with pytest.raises(TypeError):
        factories[0].params["n_estimators"] = 1


def test_neural_factories_build_the_exact_joint_24_architectures() -> None:
    config = load_paper_baselines(MODEL_CONFIG)
    lstm_factory = make_paper_factory(config, "seq2seq_lstm")
    transformer_factory = make_paper_factory(config, "transformer")

    assert isinstance(lstm_factory, TorchBaselineFactory)
    assert isinstance(transformer_factory, TorchBaselineFactory)
    lstm = lstm_factory.build_model(history_features=3, future_features=5)
    transformer = transformer_factory.build_model(history_features=3, future_features=5)

    assert isinstance(lstm, Seq2SeqLSTM)
    assert lstm.encoder.num_layers == lstm.decoder.num_layers == 2
    assert isinstance(transformer, TimeSeriesTransformer)
    assert transformer.encoder.num_layers == transformer.decoder.num_layers == 1
    assert lstm(torch.zeros(2, 168, 3), torch.zeros(2, 24, 5)).shape == (2, 24)
    assert transformer(torch.zeros(2, 168, 3), torch.zeros(2, 24, 5)).shape == (2, 24)


def test_factory_rejects_wrong_model_name_and_changed_config() -> None:
    config = load_paper_baselines(MODEL_CONFIG)

    with pytest.raises(PaperBaselineConfigError, match="paper model"):
        make_paper_factory(config, "random_forest")
    with pytest.raises(PaperBaselineConfigError, match="seeds"):
        make_paper_factory(
            replace(
                config,
                transformer=replace(config.transformer, seeds=(11, 23, 37, 41, 59)),
            ),
            "transformer",
        )


def test_validation_tail_is_the_last_61_complete_pre_evaluation_days() -> None:
    outer_train = _daily_matrix("2019-01-01", 365)

    fit_train, validation = chronological_validation_tail(outer_train, days=61)

    assert fit_train.origins.shape == (304,)
    assert validation.origins.shape == (61,)
    assert fit_train.origins[-1] == np.datetime64("2019-10-31T00:00", "ns")
    assert validation.origins[0] == np.datetime64("2019-11-01T00:00", "ns")
    assert validation.origins[-1] == np.datetime64("2019-12-31T00:00", "ns")
    assert validation.target_times.max() < np.datetime64("2020-01-01T00:00", "ns")


def test_validation_tail_rejects_non_daily_or_insufficient_training_samples() -> None:
    too_short = _daily_matrix("2019-01-01", 61)
    gapped = _daily_matrix("2019-01-01", 70)
    gapped.origins[62:] += np.timedelta64(1, "D")
    gapped.target_times[62:] += np.timedelta64(1, "D")

    with pytest.raises(ValueError, match="more than 61"):
        chronological_validation_tail(too_short, days=61)
    with pytest.raises(ValueError, match="consecutive daily"):
        chronological_validation_tail(gapped, days=61)


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
def test_boosting_factories_pass_only_the_internal_tail_for_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    class RecordingEstimator:
        instances: list[RecordingEstimator] = []

        def __init__(self, **params: object) -> None:
            self.params = params
            self.fit_kwargs: dict[str, object] = {}
            self.train_rows = 0
            RecordingEstimator.instances.append(self)

        def get_params(self, deep: bool = False) -> dict[str, object]:
            del deep
            return {
                key: None
                for key in (
                    "objective",
                    "eval_metric",
                    "n_estimators",
                    "learning_rate",
                    "max_depth",
                    "min_child_weight",
                    "subsample",
                    "colsample_bytree",
                    "reg_lambda",
                    "tree_method",
                    "early_stopping_rounds",
                    "num_leaves",
                    "min_child_samples",
                    "subsample_freq",
                    "random_state",
                    "n_jobs",
                    "verbosity",
                )
            }

        def fit(self, features: object, target: np.ndarray, **kwargs: object) -> RecordingEstimator:
            self.train_rows = len(target)
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(classical, "_estimator_class", lambda _: RecordingEstimator)
    monkeypatch.setattr(
        classical,
        "_lightgbm_early_stopping",
        lambda rounds: ("early-stopping", rounds),
        raising=False,
    )
    config = load_paper_baselines(MODEL_CONFIG)
    factory = make_paper_factory(config, model_name)
    outer_train = _daily_matrix("2019-01-01", 70)
    train, validation = chronological_validation_tail(outer_train, days=61)

    fitted = factory.fit(train, validation=validation, seed=17)

    assert len(fitted.estimators) == 24
    assert all(estimator.train_rows == 9 for estimator in fitted.estimators)
    assert all(len(estimator.fit_kwargs["eval_set"]) == 1 for estimator in fitted.estimators)
    assert all(
        len(estimator.fit_kwargs["eval_set"][0][1]) == 61
        for estimator in fitted.estimators
    )
    if model_name == "xgboost":
        assert all(
            estimator.params["early_stopping_rounds"] == 35
            for estimator in fitted.estimators
        )
    else:
        assert all(
            estimator.fit_kwargs["callbacks"] == [("early-stopping", 35)]
            for estimator in fitted.estimators
        )
