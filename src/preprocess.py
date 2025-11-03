# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Dict
from .config import cfg
from .physics import fit_pf_per_target, apply_physics_baseline

def build_physics_residuals(splits: dict) -> dict:
    pf = fit_pf_per_target(splits["train"], cfg)
    for k in ("train", "val", "test"):
        if k in splits and len(splits[k]) > 0:
            splits[k] = apply_physics_baseline(splits[k], cfg, pf)
    splits["_pf"] = pf
    return splits

def downsample(df: pd.DataFrame, orig_hz: int, target_hz: int) -> pd.DataFrame:
    if target_hz == orig_hz:
        return df.reset_index(drop=True)
    if orig_hz % target_hz != 0:
        raise ValueError(f"orig_hz({orig_hz}) not divisible by target_hz({target_hz}).")
    ratio = orig_hz // target_hz
    return df.iloc[::ratio].reset_index(drop=True)

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    v   = df.get("speed_kmh", pd.Series(0, index=df.index)).fillna(0).astype(float)
    b   = df.get("brake_bar", pd.Series(0, index=df.index)).fillna(0).astype(float)
    acc = df.get("acc_long_ms2", pd.Series(0, index=df.index)).fillna(0).astype(float)
    out = df.copy()
    out["idle"]        = (v < 0.5).astype(int)
    out["brake_event"] = (b > 0.1).astype(int)
    out["dv"]          = v.diff().fillna(0)
    out["jerk"]        = acc.diff().fillna(0)
    return out

def leak_free_split(df: pd.DataFrame, cfg) -> Dict[str, pd.DataFrame]:
    n = len(df); t0 = int(0.6 * n); t1 = int(0.8 * n)
    purge = int(cfg.purge_seconds * cfg.target_hz)
    train_end = max(t0 - purge, 0)
    val_start = min(t0 + purge, n); val_end = max(t1 - purge, 0)
    test_start = min(t1 + purge, n)
    train = df.iloc[:train_end]
    val   = df.iloc[val_start:val_end] if val_end > val_start else df.iloc[0:0]
    test  = df.iloc[test_start:]
    return {k: v.reset_index(drop=True) for k, v in {"train":train, "val":val, "test":test}.items()}

def fit_transformers(train_df: pd.DataFrame, cfg):
    scaler = RobustScaler()
    scaler.fit(train_df[cfg.feature_cols].values)
    return scaler

def apply_transformers(df: pd.DataFrame, scaler: RobustScaler, cfg):
    out = df.copy()
    out.loc[:, cfg.feature_cols] = scaler.transform(out[cfg.feature_cols].values)
    return out

def build_residual_targets(df: pd.DataFrame, cfg):
    dt = 1.0 / cfg.target_hz; T = 120.0; alpha = dt / (T + dt)
    out = df.copy()
    for col in cfg.target_cols:
        base_col = f"{col}_ema120"
        out[base_col] = out[col].ewm(alpha=alpha, adjust=False).mean()
        out[f"{col}_resid"] = out[col] - out[base_col]
    return out

def shift_ema_targets(df: pd.DataFrame, cfg):
    shift_steps = int(cfg.target_shift_s * cfg.target_hz)
    out = df.copy()
    for col in cfg.target_cols:
        out[f"{col}_resid_shift"] = out[f"{col}_resid"].shift(-shift_steps)
    if shift_steps > 0:
        out = out.iloc[:-shift_steps]
    return out.reset_index(drop=True)

def shift_phys_targets(df: pd.DataFrame, cfg):
    shift_steps = int(cfg.target_shift_s * cfg.target_hz)
    out = df.copy()
    for col in cfg.target_cols:
        rc = f"{col}_resid_phys"
        out[f"{rc}_shift"] = out[rc].shift(-shift_steps)
    if shift_steps > 0:
        out = out.iloc[:-shift_steps]
    return out.reset_index(drop=True)

def prepare_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # 1) resample + regimes
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)

    # 2) split
    splits = leak_free_split(df, cfg)

    # 3) scale features (fit on train only)
    scaler = fit_transformers(splits["train"], cfg)
    for k in ("train", "val", "test"):
        splits[k] = apply_transformers(splits[k], scaler, cfg)

    # 4) EMA residuals (optional baseline)
    for k in ("train", "val", "test"):
        splits[k] = build_residual_targets(splits[k], cfg)
        splits[k] = shift_ema_targets(splits[k], cfg)

    # 5) physics-guided residuals (recommended)
    if getattr(cfg, "use_physics_baseline", False):
        splits = build_physics_residuals(splits)       # adds *_phys and *_resid_phys
        for k in ("train", "val", "test"):
            splits[k] = shift_phys_targets(splits[k], cfg)

    # 6) === NEW: scale the final target columns actually fed to the model ===
    from .config import resolved_target_cols
    targ_cols = resolved_target_cols(cfg)  # e.g., ["PM10_ug_per_m3_resid_shift", ...] or phys variant

    t_scaler = fit_target_scaler(splits["train"], targ_cols)     # fit on TRAIN only
    for k in ("train", "val", "test"):
        splits[k] = apply_target_scaler(splits[k], targ_cols, t_scaler)

    # optionally stash for later (not required)
    splits["_target_scaler"] = t_scaler

    return splits

def fit_target_scaler(train_df: pd.DataFrame, target_cols):
    """Fit a scaler on TRAIN targets (residual target columns actually fed to the model)."""
    sc = StandardScaler()
    sc.fit(train_df[target_cols].values)
    return sc

def apply_target_scaler(df: pd.DataFrame, target_cols, sc: StandardScaler):
    """Apply the target scaler (same columns, same order) to any split."""
    out = df.copy()
    out.loc[:, target_cols] = sc.transform(out[target_cols].values)
    return out
