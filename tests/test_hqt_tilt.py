from datetime import datetime, timedelta

import numpy as np

from demand_quadratic_tilting.model import HQTResult
from demand_quadratic_tilting.tilt import tilt_from_posterior


def test_new_event_uses_one_coherent_quadratic_draw_path() -> None:
    draws = 200
    dates = [datetime(2024, 9, 16) + timedelta(hours=24 * i) for i in range(4)]
    event_id = "2024_Chuseok"
    tau_map = {(event_id, date): 24 * i for i, date in enumerate(dates)}
    event_id_of_date = {date: event_id for date in dates}
    type_of_date = {date: "Chuseok" for date in dates}
    hqt = HQTResult(
        draws_beta=np.zeros((draws, 1, 3)),
        draws_mu=np.zeros((draws, 2, 3)),
        draws_L=[
            np.repeat(np.eye(3)[None, :, :], draws, axis=0),
            np.repeat(np.eye(3)[None, :, :], draws, axis=0),
        ],
        draws_sigma_r=np.ones(draws),
        event_ids=["2023_Chuseok"],
        event_type=["Chuseok"],
        type_names=["Chuseok", "Seollal"],
        tau_map=tau_map,
        event_id_of_date=event_id_of_date,
        type_of_date=type_of_date,
        tau_unit_hours=1.0,
        tau_scale_hours=24.0,
        diagnostics={},
    )

    result, _ = tilt_from_posterior(hqt, 1.0, dates, rng_seed=2025)
    values = result["z_hat"].to_numpy()

    # A single beta draw matrix per event must yield an exact quadratic path.
    assert abs(np.diff(values, n=3)[0]) < 1e-12
