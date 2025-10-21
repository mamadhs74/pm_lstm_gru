import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Dict

def downsample(df: pd.DataFrame, orig_hz: int, target_hz: int) -> pd.DataFrame:
    if target_hz == orig_hz:
        return df.reset_index(drop=True)
    if orig_hz % target_hz != 0:
        # If not divisible, fall back to time-based resample if you have a timestamp,
        # or just return as-is to avoid silent distortion.
        # Here: safest is integer decimation only.
        raise ValueError(f"orig_hz({orig_hz}) not divisible by target_hz({target_hz}).")
    ratio = orig_hz // target_hz
    return df.iloc[::ratio].reset_index(drop=True)

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    v   = df.get("speed_kmh", pd.Series(0, index=df.index)).fillna(0).astype(float)
    b   = df.get("brake_bar", pd.Series(0, index=df.index)).fillna(0).astype(float)
    acc = df.get("acc_long_ms2", pd.Series(0, index=df.index)).fillna(0).astype(float)

    df = df.copy()
    df["idle"]        = (v < 0.5).astype(int)
    df["brake_event"] = (b > 0.1).astype(int)
    df["dv"]          = v.diff().fillna(0)
    df["jerk"]        = acc.diff().fillna(0)
    return df

def leak_free_split(df: pd.DataFrame, cfg) -> Dict[str, pd.DataFrame]:
    n = len(df)
    t0 = int(0.6 * n)
    t1 = int(0.8 * n)
    purge = int(cfg.purge_seconds * cfg.target_hz)

    # train: [0, t0 - purge)
    train_end = max(t0 - purge, 0)
    train = df.iloc[:train_end]

    # val: [t0 + purge, t1 - purge)
    val_start = min(t0 + purge, n)
    val_end   = max(t1 - purge, 0)
    if val_end <= val_start:
        val = df.iloc[0:0]  # empty
    else:
        val  = df.iloc[val_start:val_end]

    # test: [t1 + purge, n)
    test_start = min(t1 + purge, n)
    test = df.iloc[test_start:]

    return {k: v.reset_index(drop=True) for k, v in {"train":train, "val":val, "test":test}.items()}

def fit_transformers(train_df: pd.DataFrame, cfg):
    scaler = RobustScaler()
    X = train_df[cfg.feature_cols].values
    scaler.fit(X)
    return scaler

def apply_transformers(df: pd.DataFrame, scaler: RobustScaler, cfg):
    out = df.copy()
    out.loc[:, cfg.feature_cols] = scaler.transform(out[cfg.feature_cols].values)
    return out

def build_residual_targets(df: pd.DataFrame, cfg):
    dt = 1.0 / cfg.target_hz
    T = 120.0
    alpha = dt / (T + dt)
    out = df.copy()
    for col in cfg.target_cols:
        base_col = f"{col}_ema120"
        out[base_col] = out[col].ewm(alpha=alpha, adjust=False).mean()
        out[f"{col}_resid"] = out[col] - out[base_col]
    return out

def shift_targets(df: pd.DataFrame, cfg):
    shift_steps = int(cfg.target_shift_s * cfg.target_hz)
    out = df.copy()
    for col in cfg.target_cols:
        out[f"{col}_resid_shift"] = out[f"{col}_resid"].shift(-shift_steps)
    if shift_steps > 0:
        out = out.iloc[:-shift_steps]
    return out.reset_index(drop=True)
