import numpy as np
import pytest
from hqrc_v3.bayes.predictive import (
    PredictiveShapeError,
    baseline_bootstrap_draws,
    corrected_predictive_draws,
    draw_new_event_correction,
    posterior_values,
    select_posterior_indices,
    simulate_stationary_ar1,
    simulate_stationary_ar2,
    simulate_student_t_ar1,
)


def test_corrected_predictive_has_no_second_baseline_noise():
    draws = corrected_predictive_draws(
        baseline=np.array([100.0, 100.0]),
        sigma_n=10.0,
        q=np.array([[1.0, 2.0]]),
        e=np.array([[0.5, -0.5]]),
    )

    np.testing.assert_allclose(draws, np.array([[115.0, 115.0]]))


def test_predictive_draws_require_aligned_finite_trajectories():
    with pytest.raises(PredictiveShapeError, match="match baseline"):
        corrected_predictive_draws(np.array([1.0, 2.0]), 1.0, np.ones((2, 2)), np.ones((2, 3)))


def test_h0_bootstrap_keeps_horizon_blocks_joint():
    blocks = np.array([np.arange(24.0), np.arange(100.0, 124.0)])
    draws = baseline_bootstrap_draws(blocks, draws=20, seed=4)

    assert draws.shape == (20, 24)
    assert all(tuple(row) in {tuple(blocks[0]), tuple(blocks[1])} for row in draws)


def test_new_event_no_pooling_draws_training_coefficients_not_heldout():
    posterior = {"beta": np.array([[[1.0, 2.0, 3.0], [40.0, 50.0, 60.0]]])}
    values = draw_new_event_correction(
        posterior,
        holiday_type=0,
        pooling="none",
        draws=12,
        seed=7,
        training_occurrence_indices=[0],
    )

    assert values.shape == (12, 3)
    np.testing.assert_allclose(values, np.tile([1.0, 2.0, 3.0], (12, 1)))


def test_stationary_ar_is_joint_and_resets_at_event_start():
    draws = simulate_stationary_ar1(phi=np.array([0.8]), sigma=np.array([1.0]), horizon=4, seed=1)

    assert draws.shape == (1, 4)
    assert np.isfinite(draws).all()


def test_partial_pooling_uses_shared_posterior_row_and_full_covariance():
    posterior = {
        "mu": np.array([[[1.0, 1.0], [0.0, 0.0]], [[9.0, 9.0], [0.0, 0.0]]]),
        "between_cholesky": np.array(
            [
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ]
        ),
    }
    values = draw_new_event_correction(
        posterior,
        holiday_type=0,
        pooling="partial",
        draws=2,
        seed=5,
        sample_indices=np.array([0, 1]),
    )
    np.testing.assert_allclose(values, [[1.0, 1.0], [9.0, 9.0]])


def test_student_t_and_stationary_ar2_sensitivity_draws_are_finite():
    student = simulate_student_t_ar1(
        phi=np.array([0.5]), sigma=np.array([1.0]), nu=np.array([5.0]), horizon=8, seed=2
    )
    ar2 = simulate_stationary_ar2(
        ar2_phi=np.array([[0.4, 0.2]]), sigma=np.array([1.0]), horizon=8, seed=2
    )
    assert student.shape == ar2.shape == (1, 8)
    assert np.isfinite(student).all() and np.isfinite(ar2).all()


def test_scalar_chain_draw_posteriors_flatten_and_share_indices():
    posterior = {
        "mu": np.zeros((2, 3, 2, 3)),
        "gamma": np.zeros((2, 3, 2, 24)),
        "phi": np.arange(6.0).reshape(2, 3),
        "sigma_r": np.arange(10.0, 16.0).reshape(2, 3),
    }
    indices = select_posterior_indices(
        posterior, specifications={"mu": 2, "gamma": 2, "phi": 0, "sigma_r": 0}, draws=4, seed=12
    )
    np.testing.assert_allclose(
        posterior_values(posterior, "sigma_r", trailing=0, indices=indices)
        - posterior_values(posterior, "phi", trailing=0, indices=indices),
        10.0,
    )
