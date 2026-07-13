"""학습된 Seq2Seq LSTM 가중치(.pth)로 추론만 수행하는 모듈.

노트북(`src/trend_fourier_seq2seq_hqt.ipynb`)의 학습 셀 18~22 를 우회하기
위해 `infer_baseline()` 한 함수만 호출하면 train/val/test 의 baseline
예측(`*_pred_inv`)을 얻을 수 있다. 모델/데이터셋/스케일러 정의는 학습
노트북 Cell 14, 16 과 동일하다.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset


DEFAULT_FEATURE_COLS: List[str] = [
    "hm",
    "ta",
    "Fourier_Residual",
    "spring",
    "summer",
    "autoum",
    "winter",
    "is_holiday_dummies",
]
DEFAULT_TARGET_COL = "Fourier_Residual"


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================================================
# 1) 모델 정의 (노트북 Cell 14 와 동일 구조)
# =========================================================


class Encoder(nn.Module):
    def __init__(
        self, input_dim: int, hid_dim: int, n_layers: int = 1, dropout: float = 0.0
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hid_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x):
        return self.lstm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        out_len: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_len = out_len
        self.lstm = nn.LSTM(
            input_dim,
            hid_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hid_dim, 1)

    def forward(
        self,
        enc_outputs,
        states,
        y0,
        y_target=None,
        y_mask=None,
        teacher_forcing_ratio=0.0,
    ):
        h, c = states
        inp = y0.unsqueeze(1).unsqueeze(2)
        outputs = []
        for t in range(self.out_len):
            out, (h, c) = self.lstm(inp, (h, c))
            pred = self.fc(out.squeeze(1)).unsqueeze(1)
            outputs.append(pred)
            inp = pred
        return torch.cat(outputs, dim=1)


class Seq2Seq(nn.Module):
    def __init__(
        self,
        enc_input_dim: int,
        dec_input_dim: int,
        hid_dim: int,
        out_len: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder(enc_input_dim, hid_dim, n_layers, dropout)
        self.decoder = Decoder(dec_input_dim, hid_dim, out_len, n_layers, dropout)

    def forward(self, x, y0, y_target=None, y_mask=None, teacher_forcing_ratio=0.0):
        enc_out, (h, c) = self.encoder(x)
        return self.decoder(
            enc_out,
            (h, c),
            y0,
            y_target=y_target,
            y_mask=y_mask,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )


# =========================================================
# 2) Dataset (pandas 기반, 노트북 Cell 14 의 DailyStepDataset 미러)
# =========================================================


class DailyStepDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        in_days: int,
        out_days: int,
        scaler_X: StandardScaler,
        scaler_y: StandardScaler,
        exclude_empty_targets: bool = True,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.in_len = int(in_days) * 24
        self.out_len = int(out_days) * 24
        self.stride = 24

        Xs = scaler_X.transform(df[feature_cols].values)
        ys = scaler_y.transform(df[[target_col]].values).flatten()
        self.X = torch.tensor(Xs, dtype=torch.float32)
        self.y = torch.tensor(ys, dtype=torch.float32)
        self.T = len(self.X)

        starts = list(range(0, self.T - self.in_len - self.out_len + 1, self.stride))
        if exclude_empty_targets:
            self.starts = [
                s
                for s in starts
                if (s + self.in_len) < self.T and (s + self.in_len) <= self.T - 1
            ]
        else:
            self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x_seq = self.X[s : s + self.in_len]
        y_seq = self.y[s + self.in_len : s + self.in_len + self.out_len]
        y0 = self.y[s + self.in_len - 1]
        mask = torch.ones(self.out_len, dtype=torch.float32)
        return x_seq, y_seq, mask, y0


def ds_to_numpy(
    ds: DailyStepDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list, Y_list, M_list, y0_list = [], [], [], []
    for x, y, m, y0 in ds:
        X_list.append(x.numpy())
        Y_list.append(y.numpy())
        M_list.append(m.numpy())
        y0_list.append(float(y0.item()))
    return (np.stack(X_list), np.stack(Y_list), np.stack(M_list), np.array(y0_list))


# =========================================================
# 3) 추론 진입점 — 학습 산출물로 baseline 예측만 생성
# =========================================================


def load_model(
    ckpt_path: str,
    enc_input_dim: int,
    hid_dim: int,
    out_len: int,
    n_layers: int = 1,
    dropout: float = 0.0,
    device: Optional[str] = None,
) -> Seq2Seq:
    if device is None:
        device = _default_device()
    model = Seq2Seq(
        enc_input_dim=enc_input_dim,
        dec_input_dim=1,
        hid_dim=hid_dim,
        out_len=out_len,
        n_layers=n_layers,
        dropout=dropout,
    )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _infer_split(
    model: Seq2Seq,
    df_split: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    in_days: int,
    out_days: int,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    device: str,
) -> np.ndarray:
    ds = DailyStepDataset(
        df_split,
        feature_cols,
        target_col,
        in_days=in_days,
        out_days=out_days,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )
    X, _, _, y0 = ds_to_numpy(ds)
    pred = model(
        torch.from_numpy(X).float().to(device),
        torch.from_numpy(y0).float().to(device),
        teacher_forcing_ratio=0.0,
    )
    pred_np = pred.cpu().squeeze(-1).numpy()  # (N, out_len)
    pred_inv = scaler_y.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape)
    return pred_inv


def infer_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ckpt_path: str = "best_model_20260518_011512.pth",
    best_params_path: str = "best_params.json",
    scaler_X_path: str = "scaler_X.joblib",
    scaler_y_path: str = "scaler_y.joblib",
    feature_cols: Optional[List[str]] = None,
    target_col: str = DEFAULT_TARGET_COL,
    in_days: int = 7,
    out_days: int = 1,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """저장된 가중치/스케일러/하이퍼파라미터로 train/val/test baseline 예측 생성.

    Returns
    -------
    (train_pred_inv, val_pred_inv, test_pred_inv) : 각 (N, out_len*24) 형태의 역스케일 배열.
    학습 노트북 Cell 24 의 `train1["pred_inv"] = train_pred_inv.flatten()` 흐름과 호환.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    if device is None:
        device = _default_device()

    with open(best_params_path) as f:
        best_params = json.load(f)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    model = load_model(
        ckpt_path=ckpt_path,
        enc_input_dim=len(feature_cols),
        hid_dim=int(best_params["units"]),
        out_len=out_days * 24,
        n_layers=int(best_params["lstm_layers"]),
        dropout=float(best_params.get("lstm_dropout_rate", 0.0)),
        device=device,
    )

    train_pred_inv = _infer_split(
        model,
        train_df,
        feature_cols,
        target_col,
        in_days,
        out_days,
        scaler_X,
        scaler_y,
        device,
    )
    val_pred_inv = _infer_split(
        model,
        val_df,
        feature_cols,
        target_col,
        in_days,
        out_days,
        scaler_X,
        scaler_y,
        device,
    )
    test_pred_inv = _infer_split(
        model,
        test_df,
        feature_cols,
        target_col,
        in_days,
        out_days,
        scaler_X,
        scaler_y,
        device,
    )
    return train_pred_inv, val_pred_inv, test_pred_inv


# =========================================================
# 4) 레거시 polars 기반 추론 (기존 코드 호환 유지)
# =========================================================


class InferenceDataset(Dataset):
    """polars 기반 추론 데이터셋 (기존 호출자 호환용)."""

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        in_days: int,
        out_days: int,
        scaler_X: StandardScaler,
        scaler_y: StandardScaler,
        datetime_col: str = "datetime",
    ):
        self.in_len = in_days * 24
        self.out_len = out_days * 24
        self.stride = 24
        self.datetime_col = datetime_col

        X_raw = df.select(feature_cols).to_numpy()
        y_raw = df.select([target_col]).to_numpy()
        dt_raw = df[datetime_col].to_list()

        self.X = torch.tensor(scaler_X.transform(X_raw), dtype=torch.float32)
        self.y = torch.tensor(scaler_y.transform(y_raw).flatten(), dtype=torch.float32)
        self.datetimes = dt_raw
        self.T = len(self.X)
        self.starts = list(
            range(0, self.T - self.in_len - self.out_len + 1, self.stride)
        )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x_seq = self.X[s : s + self.in_len]
        y_seq = self.y[s + self.in_len : s + self.in_len + self.out_len]
        y0 = self.y[s + self.in_len - 1]
        mask = torch.ones(self.out_len, dtype=torch.float32)
        out_dts = self.datetimes[s + self.in_len : s + self.in_len + self.out_len]
        return x_seq, y_seq, mask, y0, out_dts


def predict(
    model: Seq2Seq,
    df: pl.DataFrame,
    feature_cols: List[str],
    target_col: str,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    in_days: int = 7,
    out_days: int = 1,
    batch_size: int = 64,
    datetime_col: str = "datetime",
    device: Optional[str] = None,
) -> pl.DataFrame:
    if device is None:
        device = _default_device()

    ds = InferenceDataset(
        df,
        feature_cols,
        target_col,
        in_days,
        out_days,
        scaler_X,
        scaler_y,
        datetime_col=datetime_col,
    )

    all_preds, all_trues, all_dts = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x_seq, y_seq, mask, y0, out_dts = ds[i]
            x_b = x_seq.unsqueeze(0).to(device)
            y0_b = y0.unsqueeze(0).to(device)
            pred = model(x_b, y0_b, teacher_forcing_ratio=0.0)
            pred_np = pred.cpu().squeeze(-1).numpy()
            all_preds.append(pred_np[0])
            all_trues.append(y_seq.numpy())
            all_dts.append(out_dts)

    preds_arr = np.concatenate(all_preds)
    trues_arr = np.concatenate(all_trues)
    preds_inv = scaler_y.inverse_transform(preds_arr.reshape(-1, 1)).flatten()
    trues_inv = scaler_y.inverse_transform(trues_arr.reshape(-1, 1)).flatten()
    dts_flat = [dt for dts in all_dts for dt in dts]

    return pl.DataFrame(
        {
            datetime_col: dts_flat,
            "y_true": trues_inv,
            "y_pred": preds_inv,
        }
    )


def predict_batch(
    model: Seq2Seq,
    X: np.ndarray,
    y0: np.ndarray,
    scaler_y: StandardScaler,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    if device is None:
        device = _default_device()
    X_t = torch.from_numpy(X).float()
    y0_t = torch.from_numpy(y0).float()
    loader = DataLoader(TensorDataset(X_t, y0_t), batch_size=batch_size)

    preds_list = []
    model.eval()
    with torch.no_grad():
        for xb, y0b in loader:
            xb, y0b = xb.to(device), y0b.to(device)
            pred = model(xb, y0b, teacher_forcing_ratio=0.0)
            preds_list.append(pred.cpu().squeeze(-1).numpy())
    preds_all = np.concatenate(preds_list, axis=0)
    return scaler_y.inverse_transform(preds_all.reshape(-1, 1)).reshape(preds_all.shape)
