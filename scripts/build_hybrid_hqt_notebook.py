"""Build the canonical HP–Fourier–Seq2Seq-LSTM + quadratic HQT notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "ver2" / "hybrid_hqt" / "hqt_verification_source.ipynb"


def _markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def _code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> nbformat.NotebookNode:
    """Return a deterministic, output-free canonical notebook."""

    cells = [
        _markdown(
            r"""
            # HP–Fourier–Seq2Seq-LSTM + Hierarchical Quadratic Tilting

            이 노트북은 저장된 Seq2Seq-LSTM 체크포인트로 하이브리드 베이스라인을
            재현하고, 명절 잔차에 계층 이차 틸트를 적용한다.

            ```text
            baseline = HP-filter trend
                     + Fourier seasonality
                     + Seq2Seq-LSTM forecast of the Fourier residual

            z(i,t) = beta(i,0) + beta(i,1) * tau + beta(i,2) * tau^2 + error
            ```

            대체공휴일에 대한 별도 점프 절편은 두지 않는다. 설날·추석 이벤트별
            이차계수는 유형별 사후분포에서 계층적으로 pooling한다.

            - 입력 윈도: 168시간
            - 예측 구간: 24시간
            - train: 2019–2022
            - validation: 2023
            - test: 2024-01-08~2024-10-31 (split별 LSTM 윈도 구성으로 168시간 제외)

            > 재현 주의: 기존 하이브리드 노트북과 동일하게 HP trend를 전체
            > 2019–2024 수요에서 한 번에 추출한다. 따라서 이 결과는 leakage-safe한
            > 다중 베이스라인 ver2 결과와 직접 비교하지 않는다.
            """
        ),
        _markdown("## 0. Setup"),
        _code(
            """
            import json
            import pathlib
            import sys
            import warnings
            from itertools import product

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import polars as pl
            import torch
            from scipy.stats import norm
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import (
                mean_absolute_percentage_error,
                mean_squared_error,
                r2_score,
            )
            from statsmodels.tsa.filters.hp_filter import hpfilter

            _candidates = [pathlib.Path.cwd().resolve(), pathlib.Path.cwd().resolve().parent]
            PROJECT_ROOT = next(
                path for path in _candidates if (path / "power_demand_final.csv").exists()
            )
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from demand_quadratic_tilting.constants import CHUSEOK_LABELS, SEOLLAL_LABELS
            from demand_quadratic_tilting.inference import infer_baseline
            from demand_quadratic_tilting.model import (
                compute_sigma_and_residuals,
                fit_hqt_pymc_lkj,
            )
            from demand_quadratic_tilting.tilt import apply_tilt, tilt_from_posterior
            from demand_quadratic_tilting.windows import build_holiday_windows_and_tau

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            DATA_PATH = PROJECT_ROOT / "power_demand_final.csv"
            MODEL_ASSETS = PROJECT_ROOT / "ver2" / "hybrid_hqt" / "model_assets"
            CKPT_PATH = MODEL_ASSETS / "best_model_20260518_011512.pth"
            OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "hybrid_hqt_notebook_quadratic"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            if torch.cuda.is_available():
                INFERENCE_DEVICE = "cuda"
            elif torch.backends.mps.is_available():
                INFERENCE_DEVICE = "mps"
            else:
                INFERENCE_DEVICE = "cpu"

            print(f"project root     : {PROJECT_ROOT}")
            print(f"data             : {DATA_PATH}")
            print(f"checkpoint       : {CKPT_PATH}")
            print(f"inference device : {INFERENCE_DEVICE}")
            print(f"output           : {OUTPUT_DIR}")
            """
        ),
        _markdown("## 1. 데이터 로드"),
        _code(
            """
            df = pd.read_csv(DATA_PATH)
            df["일시"] = pd.to_datetime(df["일시"])
            df["holiday_name"] = df["holiday_name"].fillna("non-event")
            df = df.sort_values("일시").reset_index(drop=True)

            print(f"총 {len(df):,}행 / {len(df.columns)}개 컬럼")
            print(f"기간: {df['일시'].min()} ~ {df['일시'].max()}")
            """
        ),
        _markdown("## 2. HP-filter 추세 추출"),
        _code(
            """
            LAMBDA_HP = 1.28e8

            y_all_raw = df["power demand(MW)"].to_numpy()
            _, trend_all = hpfilter(y_all_raw, lamb=LAMBDA_HP)
            df["trend"] = trend_all
            df["detrend"] = df["power demand(MW)"] - df["trend"]

            print(f"trend range: {trend_all.min():,.1f} ~ {trend_all.max():,.1f} MW")
            print(f"detrended std: {df['detrend'].std():,.1f} MW")
            """
        ),
        _markdown("## 3. Fourier 계절성 분해 — 2023 validation 선택"),
        _code(
            """
            def generate_fourier_terms(timesteps, period, harmonics):
                terms = []
                for k in range(1, harmonics + 1):
                    terms.append(np.sin(2 * np.pi * k * timesteps / period))
                    terms.append(np.cos(2 * np.pi * k * timesteps / period))
                return np.stack(terms, axis=1)


            y_all = df["detrend"].to_numpy()
            t_all = np.arange(len(y_all))
            train_end = int((df["일시"] <= "2022-12-31 23:00:00").sum())
            val_end = int((df["일시"] <= "2023-12-31 23:00:00").sum())

            best_mse = float("inf")
            best_model = None
            best_X_all = None
            best_K_combo = None

            for daily_k, weekly_k, yearly_k in product(
                [1, 2, 3], range(1, 9), [1, 2, 3, 4]
            ):
                X_all = np.hstack(
                    [
                        generate_fourier_terms(t_all, 24, daily_k),
                        generate_fourier_terms(t_all, 24 * 7, weekly_k),
                        generate_fourier_terms(t_all, 24 * 365.25, yearly_k),
                    ]
                )
                model = LinearRegression().fit(X_all[:train_end], y_all[:train_end])
                mse = mean_squared_error(
                    y_all[train_end:val_end], model.predict(X_all[train_end:val_end])
                )
                if mse < best_mse:
                    best_mse = mse
                    best_model = model
                    best_X_all = X_all
                    best_K_combo = (daily_k, weekly_k, yearly_k)

            seasonality_pred = best_model.predict(best_X_all)
            df["Fourier_Seasonality"] = seasonality_pred
            df["Fourier_Residual"] = y_all - seasonality_pred

            print(f"best K (daily, weekly, yearly): {best_K_combo}")
            print(f"validation MSE: {best_mse:,.2f} MW²")
            """
        ),
        _markdown("## 4. Train / Validation / Test 분할"),
        _code(
            """
            train = df[df["일시"] <= "2022-12-31 23:00:00"].copy()
            validation = df[
                (df["일시"] > "2022-12-31 23:00:00")
                & (df["일시"] <= "2023-12-31 23:00:00")
            ].copy()
            test = df[df["일시"] > "2023-12-31 23:00:00"].copy()

            for frame in (train, validation, test):
                frame.set_index("일시", inplace=True)
                frame.sort_index(inplace=True)

            print(f"train      : {len(train):>6,}h ({train.index.min()} ~ {train.index.max()})")
            print(
                f"validation : {len(validation):>6,}h "
                f"({validation.index.min()} ~ {validation.index.max()})"
            )
            print(f"test       : {len(test):>6,}h ({test.index.min()} ~ {test.index.max()})")
            """
        ),
        _markdown("## 5. 저장된 Seq2Seq-LSTM으로 하이브리드 베이스라인 추론"),
        _code(
            """
            feature_cols = [
                "hm",
                "ta",
                "Fourier_Residual",
                "spring",
                "summer",
                "autoum",
                "winter",
                "is_holiday_dummies",
            ]

            train_pred_inv, val_pred_inv, test_pred_inv = infer_baseline(
                train_df=train,
                val_df=validation,
                test_df=test,
                ckpt_path=CKPT_PATH,
                best_params_path=MODEL_ASSETS / "best_params.json",
                scaler_X_path=MODEL_ASSETS / "scaler_X.joblib",
                scaler_y_path=MODEL_ASSETS / "scaler_y.joblib",
                feature_cols=feature_cols,
                target_col="Fourier_Residual",
                in_days=7,
                out_days=1,
                device=INFERENCE_DEVICE,
            )

            train1 = train.iloc[-train_pred_inv.size :].copy()
            val1 = validation.iloc[-val_pred_inv.size :].copy()
            test1 = test.iloc[-test_pred_inv.size :].copy()

            for frame, prediction in (
                (train1, train_pred_inv),
                (val1, val_pred_inv),
                (test1, test_pred_inv),
            ):
                frame["pred_inv"] = prediction.ravel()
                frame["hybrid"] = (
                    frame["trend"] + frame["Fourier_Seasonality"] + frame["pred_inv"]
                )

            print("baseline ready")
            print(f"train1={len(train1):,}h, val1={len(val1):,}h, test1={len(test1):,}h")
            """
        ),
        _markdown("## 6. H0 하이브리드 베이스라인 평가"),
        _code(
            """
            y_true = test1["power demand(MW)"].to_numpy()
            y_hat = test1["hybrid"].to_numpy()

            print("=== H0 baseline — full test ===")
            print(f"MAE  : {np.mean(np.abs(y_hat - y_true)):,.3f} MW")
            print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_hat)):,.3f} MW")
            print(f"R²   : {r2_score(y_true, y_hat):.5f}")
            """
        ),
        _markdown(
            r"""
            ## 7. Hierarchical Quadratic Tilting

            비명절 잔차의 표준편차로 잔차를 표준화하고, 이벤트 내 상대시간을
            일 단위로 변환해 다음 모형을 적합한다.

            $$
            z_{i,t} \sim \mathcal{N}(\beta_{i0}+\beta_{i1}\tau+\beta_{i2}\tau^2,\sigma_r^2)
            $$

            $$
            \beta_i = \mu_{h(i)} + L_{h(i)}\epsilon_i,\qquad
            \epsilon_i\sim\mathcal{N}(0,I)
            $$

            유형별 공분산에는 LKJ prior를 사용한다. 이벤트 윈도는 공식 3일과
            앞뒤 1일을 합친 5일(120시간)로 통일한다. 2024년 이벤트는 학습에서
            보지 않은 신규 이벤트이며, 하나의 사후 이차곡선을 이벤트 전체에서
            공유한다.
            """
        ),
        _code(
            """
            def _to_hqt_frame(frame):
                return pl.DataFrame(
                    {
                        "datetime": frame.index.to_pydatetime().tolist(),
                        "power demand(MW)": frame["power demand(MW)"].to_numpy(),
                        "hybrid": frame["hybrid"].to_numpy(),
                        "holiday_name": frame["holiday_name"].astype(str).tolist(),
                    }
                )


            train_hqt = _to_hqt_frame(train1)
            val_hqt = _to_hqt_frame(val1)
            test_hqt = _to_hqt_frame(test1)
            all_hqt = pl.concat([train_hqt, val_hqt, test_hqt]).sort("datetime")

            # 별도 점프 절편을 쓰지 않으므로 대체공휴일 라벨이 윈도 경계를
            # 늘리지 않게 하고, 공식 3일의 앞뒤 1일 패딩 안에서 곡선으로 처리한다.
            window_input = all_hqt.with_columns(
                pl.when(pl.col("holiday_name").str.starts_with("Alternative holiday"))
                .then(pl.lit("non-event"))
                .otherwise(pl.col("holiday_name"))
                .alias("holiday_name")
            )
            (
                holiday_windows,
                tau_map,
                event_id_of_date,
                type_of_date,
                tau_unit_hours,
            ) = build_holiday_windows_and_tau(
                window_input,
                pre_pad_days=1,
                post_pad_days=1,
            )

            sigma_resid, train_residuals = compute_sigma_and_residuals(
                train_hqt,
                y_col="power demand(MW)",
                pred_col="hybrid",
            )
            print(f"non-holiday residual sigma: {sigma_resid:,.2f} MW")
            """
        ),
        _markdown("## 8. HQT 적합 — 2019–2022 train 이벤트만 사용"),
        _code(
            """
            HQT_CHAINS = 4
            HQT_DRAWS = 5000
            HQT_TUNE = 3000
            HQT_TARGET_ACCEPT = 0.99
            HQT_RANDOM_SEED = 2025

            hqt = fit_hqt_pymc_lkj(
                df_train=train_hqt,
                sigma_resid=sigma_resid,
                residuals=train_residuals,
                tau_map=tau_map,
                event_id_of_date=event_id_of_date,
                type_of_date=type_of_date,
                chains=HQT_CHAINS,
                draws=HQT_DRAWS,
                tune=HQT_TUNE,
                target_accept=HQT_TARGET_ACCEPT,
                random_seed=HQT_RANDOM_SEED,
                sampler="nuts",
                tau_unit_hours=tau_unit_hours,
                tau_scale_hours=24.0,
            )

            print(f"fit events: {hqt.event_ids}")
            print(f"diagnostics: {hqt.diagnostics}")
            """
        ),
        _markdown("## 9. 2024 신규 이벤트에 posterior quadratic tilt 적용"),
        _code(
            """
            test_tilt, half_width_const_90 = tilt_from_posterior(
                hqt,
                sigma_resid,
                test_hqt["datetime"].to_list(),
                tilt_mode="hybrid",
                ci=0.90,
                rng_seed=HQT_RANDOM_SEED,
            )
            test_adjusted = apply_tilt(test_hqt, test_tilt, baseline_col="hybrid")

            hybrid_tilt = pd.DataFrame(
                {
                    "y": test_adjusted["power demand(MW)"].to_numpy(),
                    "baseline_pred": test_adjusted["hybrid"].to_numpy(),
                    "e_tilt": test_adjusted["e_tilt"].to_numpy(),
                    "tilted_pred": test_adjusted["tilted_pred"].to_numpy(),
                    "half_width_t": test_adjusted["half_width_t"].to_numpy(),
                },
                index=pd.DatetimeIndex(test_adjusted["datetime"].to_list(), name="datetime"),
            )
            print(f"active event hours: {(hybrid_tilt['half_width_t'] > 0).sum():,}")
            """
        ),
        _markdown("## 10. H0 / H1 / H2 점예측 비교"),
        _code(
            """
            train_resid = train1["power demand(MW)"] - train1["hybrid"]
            type_mean = {}
            for event_type in hqt.type_names:
                stamps = [
                    pd.Timestamp(ts)
                    for ts, name in type_of_date.items()
                    if name == event_type and pd.Timestamp(ts) in train1.index
                ]
                type_mean[event_type] = float(train_resid.loc[stamps].mean())

            test_h1 = test1["hybrid"].copy()
            for event_type, correction in type_mean.items():
                stamps = [
                    pd.Timestamp(ts)
                    for ts, name in type_of_date.items()
                    if name == event_type and pd.Timestamp(ts) in test1.index
                ]
                test_h1.loc[stamps] += correction

            def _point_row(name, actual, prediction):
                actual = np.asarray(actual, dtype=float)
                prediction = np.asarray(prediction, dtype=float)
                error = prediction - actual
                return {
                    "model": name,
                    "n": int(actual.size),
                    "MAE (MW)": float(np.mean(np.abs(error))),
                    "RMSE (MW)": float(np.sqrt(np.mean(error**2))),
                    "WAPE (%)": float(100 * np.sum(np.abs(error)) / np.sum(np.abs(actual))),
                    "MAPE": float(mean_absolute_percentage_error(actual, prediction)),
                    "bias (MW)": float(np.mean(error)),
                    "R²": float(r2_score(actual, prediction)),
                }

            event_test_idx = pd.DatetimeIndex(
                [ts for ts in test1.index if ts.to_pydatetime() in event_id_of_date]
            )
            full_tbl = pd.DataFrame(
                [
                    _point_row("H0 baseline", test1["power demand(MW)"], test1["hybrid"]),
                    _point_row("H1 type-mean", test1["power demand(MW)"], test_h1),
                    _point_row(
                        "H2 quadratic HQT",
                        hybrid_tilt["y"],
                        hybrid_tilt["tilted_pred"],
                    ),
                ]
            )
            event_tbl = pd.DataFrame(
                [
                    _point_row(
                        "H0 baseline",
                        test1.loc[event_test_idx, "power demand(MW)"],
                        test1.loc[event_test_idx, "hybrid"],
                    ),
                    _point_row(
                        "H1 type-mean",
                        test1.loc[event_test_idx, "power demand(MW)"],
                        test_h1.loc[event_test_idx],
                    ),
                    _point_row(
                        "H2 quadratic HQT",
                        hybrid_tilt.loc[event_test_idx, "y"],
                        hybrid_tilt.loc[event_test_idx, "tilted_pred"],
                    ),
                ]
            )

            print("=== Full test ===")
            print(full_tbl.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
            print("\\n=== Event windows ===")
            print(event_tbl.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
            """
        ),
        _markdown("## 11. 90% prediction interval calibration"),
        _code(
            """
            ALPHA = 0.10
            zcrit = norm.ppf(1 - ALPHA / 2)
            preds = hybrid_tilt.copy()
            preds["h1_pred"] = test_h1.reindex(preds.index)

            half_base = zcrit * sigma_resid
            half_h2 = preds["half_width_t"].where(preds["half_width_t"] > 0, half_base)
            preds["lower_90_h0"] = preds["baseline_pred"] - half_base
            preds["upper_90_h0"] = preds["baseline_pred"] + half_base
            preds["lower_90_h1"] = preds["h1_pred"] - half_base
            preds["upper_90_h1"] = preds["h1_pred"] + half_base
            preds["lower_90_h2"] = preds["tilted_pred"] - half_h2
            preds["upper_90_h2"] = preds["tilted_pred"] + half_h2

            def crps_gaussian(y, mu, sigma):
                score = (y - mu) / sigma
                return sigma * (
                    score * (2 * norm.cdf(score) - 1)
                    + 2 * norm.pdf(score)
                    - 1 / np.sqrt(np.pi)
                )

            def _interval_row(scope, model, y, lo, hi, mu, sigma):
                y = np.asarray(y, dtype=float)
                lo = np.asarray(lo, dtype=float)
                hi = np.asarray(hi, dtype=float)
                mu = np.asarray(mu, dtype=float)
                sigma = np.asarray(sigma, dtype=float)
                return {
                    "set": scope,
                    "model": model,
                    "n": int(y.size),
                    "PICP@90%": float(100 * np.mean((y >= lo) & (y <= hi))),
                    "MPIW (MW)": float(np.mean(hi - lo)),
                    "CRPS": float(np.mean(crps_gaussian(y, mu, sigma))),
                }

            y_vec = preds["y"].to_numpy()
            event_mask = preds["half_width_t"].to_numpy() > 0
            sigma_h0 = np.full(len(preds), sigma_resid)
            sigma_h2 = half_h2.to_numpy() / zcrit
            rows = []
            for scope, mask in (
                ("Full", np.ones(len(preds), dtype=bool)),
                ("Event", event_mask),
            ):
                rows.extend(
                    [
                        _interval_row(
                            scope,
                            "baseline (H0)",
                            y_vec[mask],
                            preds["lower_90_h0"].to_numpy()[mask],
                            preds["upper_90_h0"].to_numpy()[mask],
                            preds["baseline_pred"].to_numpy()[mask],
                            sigma_h0[mask],
                        ),
                        _interval_row(
                            scope,
                            "type-mean (H1)",
                            y_vec[mask],
                            preds["lower_90_h1"].to_numpy()[mask],
                            preds["upper_90_h1"].to_numpy()[mask],
                            preds["h1_pred"].to_numpy()[mask],
                            sigma_h0[mask],
                        ),
                        _interval_row(
                            scope,
                            "quadratic HQT (H2)",
                            y_vec[mask],
                            preds["lower_90_h2"].to_numpy()[mask],
                            preds["upper_90_h2"].to_numpy()[mask],
                            preds["tilted_pred"].to_numpy()[mask],
                            sigma_h2[mask],
                        ),
                    ]
                )

            interval_tbl = pd.DataFrame(rows)
            print(interval_tbl.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
            """
        ),
        _markdown("## 12. Posterior type-level quadratic curves"),
        _code(
            """
            tau_grid = np.linspace(-2, 2, 193)
            curve_design = np.column_stack([np.ones_like(tau_grid), tau_grid, tau_grid**2])

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            for type_index, (ax, event_type) in enumerate(zip(axes, hqt.type_names)):
                curve_draws = sigma_resid * (hqt.draws_mu[:, type_index, :] @ curve_design.T)
                lower, median, upper = np.percentile(curve_draws, [5, 50, 95], axis=0)
                ax.fill_between(tau_grid, lower, upper, color="tab:blue", alpha=0.2)
                ax.plot(tau_grid, median, color="tab:blue", linewidth=2)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
                ax.set_title(event_type)
                ax.set_xlabel("relative day (tau)")
                ax.grid(alpha=0.25)
            axes[0].set_ylabel("posterior tilt (MW)")
            fig.suptitle("Type-level hierarchical quadratic tilt — posterior 90% interval")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "hqt_type_quadratic_curves.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        _markdown("## 13. 2024 설날·추석 보정 결과"),
        _code(
            """
            def _plot_holiday(ax, sub, title):
                active = sub["half_width_t"] > 0
                if active.any():
                    ax.axvspan(
                        sub.index[active][0],
                        sub.index[active][-1],
                        color="lightcoral",
                        alpha=0.08,
                        label="event window",
                    )
                ax.fill_between(
                    sub.index,
                    sub["lower_90_h2"],
                    sub["upper_90_h2"],
                    color="lightsteelblue",
                    alpha=0.45,
                    label="HQT 90% PI",
                )
                ax.plot(sub.index, sub["y"], color="black", linewidth=1.6, label="realized")
                ax.plot(
                    sub.index,
                    sub["baseline_pred"],
                    color="tab:orange",
                    linewidth=1.3,
                    label="HP–Fourier–LSTM (H0)",
                )
                ax.plot(
                    sub.index,
                    sub["tilted_pred"],
                    color="tab:blue",
                    linewidth=1.5,
                    label="quadratic HQT (H2)",
                )
                ax.set_title(title)
                ax.set_ylabel("demand (MW)")
                ax.grid(alpha=0.25)
                ax.legend(loc="upper left", fontsize=8)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            _plot_holiday(axes[0], preds.loc["2024-02-08":"2024-02-13"], "Seollal 2024")
            _plot_holiday(axes[1], preds.loc["2024-09-14":"2024-09-19"], "Chuseok 2024")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "hqt_2024_holidays.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        _markdown("## 14. H0 / H1 / H2 이벤트 비교"),
        _code(
            """
            def _plot_three_way(ax, sub, title):
                active = sub["half_width_t"] > 0
                if active.any():
                    ax.axvspan(
                        sub.index[active][0],
                        sub.index[active][-1],
                        color="lightcoral",
                        alpha=0.08,
                    )
                ax.plot(sub.index, sub["y"], color="black", linewidth=1.6, label="realized")
                ax.plot(
                    sub.index,
                    sub["baseline_pred"],
                    color="tab:orange",
                    linewidth=1.3,
                    label="H0 baseline",
                )
                ax.plot(
                    sub.index,
                    sub["h1_pred"],
                    color="tab:purple",
                    linewidth=1.3,
                    linestyle=":",
                    label="H1 type mean",
                )
                ax.plot(
                    sub.index,
                    sub["tilted_pred"],
                    color="tab:blue",
                    linewidth=1.6,
                    label="H2 quadratic HQT",
                )
                ax.set_title(title)
                ax.set_ylabel("demand (MW)")
                ax.grid(alpha=0.25)
                ax.legend(loc="upper left", fontsize=8)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            _plot_three_way(axes[0], preds.loc["2024-02-08":"2024-02-13"], "Seollal 2024")
            _plot_three_way(axes[1], preds.loc["2024-09-14":"2024-09-19"], "Chuseok 2024")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "hqt_h0_h1_h2_comparison.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        _markdown("## 15. 결과 저장"),
        _code(
            """
            def _metric_summary(actual, prediction):
                actual = np.asarray(actual, dtype=float)
                prediction = np.asarray(prediction, dtype=float)
                error = prediction - actual
                return {
                    "n": int(actual.size),
                    "mae_mw": float(np.mean(np.abs(error))),
                    "rmse_mw": float(np.sqrt(np.mean(error**2))),
                    "wape_pct": float(100 * np.sum(np.abs(error)) / np.sum(np.abs(actual))),
                    "smape_pct": float(
                        100
                        * np.mean(
                            2 * np.abs(error) / (np.abs(actual) + np.abs(prediction))
                        )
                    ),
                    "bias_mw": float(np.mean(error)),
                }

            output_predictions = preds.copy()
            output_predictions["holiday_name"] = test1["holiday_name"].reindex(preds.index)
            output_predictions["event_id"] = [
                event_id_of_date.get(ts.to_pydatetime()) for ts in preds.index
            ]
            output_predictions["event_type"] = [
                type_of_date.get(ts.to_pydatetime()) for ts in preds.index
            ]
            output_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index_label="datetime")

            full_tbl.to_csv(OUTPUT_DIR / "point_metrics_full.csv", index=False)
            event_tbl.to_csv(OUTPUT_DIR / "point_metrics_event_window.csv", index=False)
            interval_tbl.to_csv(OUTPUT_DIR / "interval_metrics.csv", index=False)

            coefficient_rows = []
            coefficient_names = ["intercept", "tau", "tau_squared"]
            for type_index, event_type in enumerate(hqt.type_names):
                for coefficient_index, coefficient_name in enumerate(coefficient_names):
                    values_mw = sigma_resid * hqt.draws_mu[:, type_index, coefficient_index]
                    lower, median, upper = np.percentile(values_mw, [5, 50, 95])
                    coefficient_rows.append(
                        {
                            "event_type": event_type,
                            "coefficient": coefficient_name,
                            "median_mw": float(median),
                            "p05_mw": float(lower),
                            "p95_mw": float(upper),
                        }
                    )
            coefficient_tbl = pd.DataFrame(coefficient_rows)
            coefficient_tbl.to_csv(OUTPUT_DIR / "type_curve_posterior_mw.csv", index=False)

            posterior_arrays = {
                "draws_beta": hqt.draws_beta,
                "draws_mu": hqt.draws_mu,
                "draws_sigma_r": hqt.draws_sigma_r,
            }
            for type_index, draws in enumerate(hqt.draws_L):
                posterior_arrays[f"draws_L_{type_index}"] = draws
            np.savez_compressed(OUTPUT_DIR / "posterior_draws.npz", **posterior_arrays)

            official_labels = CHUSEOK_LABELS | SEOLLAL_LABELS
            official_mask = np.asarray(test1["holiday_name"].isin(official_labels))
            actual = preds["y"].to_numpy()
            baseline = preds["baseline_pred"].to_numpy()
            tilted = preds["tilted_pred"].to_numpy()

            summary = {
                "config": {
                    "method": "hierarchical_quadratic_tilting",
                    "separate_alternative_holiday_jump": False,
                    "chains": HQT_CHAINS,
                    "draws": HQT_DRAWS,
                    "tune": HQT_TUNE,
                    "target_accept": HQT_TARGET_ACCEPT,
                    "random_seed": HQT_RANDOM_SEED,
                    "hp_lambda": LAMBDA_HP,
                    "history_hours": 168,
                    "horizon_hours": 24,
                    "inference_device": INFERENCE_DEVICE,
                },
                "data": {
                    "train_hours": int(len(train1)),
                    "validation_hours": int(len(val1)),
                    "test_hours": int(len(test1)),
                    "test_start": str(test1.index.min()),
                    "test_end": str(test1.index.max()),
                    "hp_filter_full_sample": True,
                    "fourier_best_k": [int(value) for value in best_K_combo],
                    "fourier_validation_mse": float(best_mse),
                },
                "sigma_resid_mw": float(sigma_resid),
                "fit_event_ids": list(hqt.event_ids),
                "diagnostics": hqt.diagnostics,
                "metrics": {
                    "overall_baseline": _metric_summary(actual, baseline),
                    "overall_tilted": _metric_summary(actual, tilted),
                    "event_window_baseline": _metric_summary(
                        actual[event_mask], baseline[event_mask]
                    ),
                    "event_window_tilted": _metric_summary(
                        actual[event_mask], tilted[event_mask]
                    ),
                    "official_holiday_baseline": _metric_summary(
                        actual[official_mask], baseline[official_mask]
                    ),
                    "official_holiday_tilted": _metric_summary(
                        actual[official_mask], tilted[official_mask]
                    ),
                },
            }
            (OUTPUT_DIR / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"saved notebook artifacts to {OUTPUT_DIR}")
            """
        ),
    ]

    for index, cell in enumerate(cells):
        cell["id"] = f"hybrid-hqt-{index:02d}"

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return notebook


def write_notebook(output_path: Path = DEFAULT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(build_notebook(), handle)
    return output_path


if __name__ == "__main__":
    print(write_notebook())
