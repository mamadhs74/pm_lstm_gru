# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Dict
from .config import cfg
from .physics import fit_pf_per_target, apply_physics_baseline
from pathlib import Path
import pickle

def _ema(series: pd.Series, half_life_s: float, hz: int) -> pd.Series:
    alpha = 1.0 - np.exp(-np.log(2)/ (half_life_s * hz))
    return series.ewm(alpha=alpha, adjust=False).mean()

def canonicalize_units_and_names(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Map your raw CSV's German column names to the canonical names your pipeline expects,
    and normalize units to SI (notably: lateral accel in g -> m/s^2).
    This is a pure 'add columns (and unit convert) if present' step; it won't break
    files that already use canonical names.
    """
    out = df.copy()

    # 1) Straight renames / copies (if source exists and target not present)
    colmap = {
        "Velocity_[km\\h]":               "speed_kmh",
        "BR8_Laengsbeschl_[m\\s2]":       "acc_long_ms2",
        "BR5_Bremsdruck_[bar]":           "brake_bar",
        "BR5_Giergeschw_[Grad\\s]":       "yaw_deg_s",
        "LW1_Lenk_Gesch_[Grad\\s]":       "steer_rate_deg_s",
        "MO1_Drehzahl_[1\\min]":          "rpm",
        "KO2_Aussen_T_[Grad_Celsius]":    "ambient_temp_c",
        "BR3_Rad_VL_[km\\h]":             "wheel_fl_kmh",  # vorne links
        "BR3_Rad_VR_[km\\h]":             "wheel_fr_kmh",  # vorne rechts
        "BR3_Rad_HL_[km\\h]":             "wheel_rl_kmh",  # hinten links
        "BR3_Rad_HR_[km\\h]":             "wheel_rr_kmh",  # hinten rechts
        "Altitude_WGS_84_[m]":            "altitude_m",
        "Distance_[m]":                   "distance_m",
        "X_Achse_[s]":                    "time_s",
        # If you ever need UTC timestamps instead, you can derive time_s from there.
    }
    for src, dst in colmap.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src].astype(float)

    # 2) Lateral acceleration: convert g -> m/s^2 into acc_lat_ms2
    if "acc_lat_ms2" not in out.columns:
        if "BR2_Querbeschl_[g]" in out.columns:
            out["acc_lat_ms2"] = out["BR2_Querbeschl_[g]"].astype(float) * float(getattr(cfg, "g", 9.80665))
        elif "acc_lat_g" in out.columns:
            out["acc_lat_ms2"] = out["acc_lat_g"].astype(float) * float(getattr(cfg, "g", 9.80665))
        # else: leave missing; your later code will fill NaN/0 or raise if truly required

    return out

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
    ax  = df.get("acc_long_ms2", pd.Series(0, index=df.index)).fillna(0).astype(float)

    out = df.copy()
    out["idle"]        = (v < 0.5).astype(int)
    out["brake_event"] = (b > 0.1).astype(int)
    out["dv"]          = v.diff().fillna(0)
    out["jerk"]        = ax.diff().fillna(0)

    # NEW: brake pressure rate (per second; your sampling becomes target_hz later)
    out["brake_bar_rate"] = b.diff().fillna(0) * cfg.orig_hz

    # NEW: short EMA of longitudinal acceleration to stabilize impulses (half-life ~0.6 s)
    out["acc_long_ema"] = _ema(ax, half_life_s=0.6, hz=cfg.orig_hz)

    # NEW: longitudinal load-transfer proxy ΔFz_front ≈ m * ax * h_CG / L (scaled)
    m   = float(cfg.vehicle_mass_kg)
    Lwb = float(cfg.wheelbase_m)
    hcg = float(getattr(cfg, "cg_height_m", 0.62))
    out["load_transfer_proxy"] = (m * ax * hcg / max(Lwb, 1e-6)) / max(m * 9.81, 1e-6)  # scale by weight

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
    dt = 1.0 / cfg.target_hz
    T = 2
    alpha = dt / (T + dt)
    out = df.copy()
    for col in cfg.target_cols:
        ema_col = f"{col}_ema120"
        resid_col = f"{col}_resid"
        out[ema_col] = out[col].ewm(alpha=alpha, adjust=False).mean()
        out[resid_col] = (out[col] - out[ema_col]) / (np.abs(out[ema_col]) + 1e-6)
    return out

def shift_ema_targets(df: pd.DataFrame, cfg):
    shift_steps = int(cfg.target_shift_s * cfg.target_hz)
    out = df.copy()

    for col in cfg.target_cols:
        ema_col = f"{col}_ema120"
        resid_col = f"{col}_resid"

        out[f"{col}_shift"] = out[col].shift(-shift_steps)
        out[f"{ema_col}_shift"] = out[ema_col].shift(-shift_steps)
        out[f"{resid_col}_shift"] = out[resid_col].shift(-shift_steps)

    if shift_steps > 0:
        out = out.iloc[:-shift_steps]

    return out.reset_index(drop=True)

def shift_phys_targets(df: pd.DataFrame, cfg):
    shift_steps = int(cfg.target_shift_s * cfg.target_hz)
    out = df.copy()

    for col in cfg.target_cols:
        short = col.split("_ug")[0] if "_ug" in col else col
        rc = f"{col}_resid_phys"
        pc = f"{short}_phys"

        out[f"{rc}_shift"] = out[rc].shift(-shift_steps)
        out[f"{pc}_shift"] = out[pc].shift(-shift_steps)
        out[f"{col}_shift"] = out[col].shift(-shift_steps)

    if shift_steps > 0:
        out = out.iloc[:-shift_steps]

    return out.reset_index(drop=True)
    
def prepare_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # 0) normalize units and create canonical columns
    df = canonicalize_units_and_names(df, cfg)

    # 1) resample + regime features
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)

    # 2) split first, but DO NOT scale before physics
    splits = leak_free_split(df, cfg)

    # 3) build EMA residual targets on raw physical values
    for k in ("train", "val", "test"):
        splits[k] = build_residual_targets(splits[k], cfg)
        splits[k] = shift_ema_targets(splits[k], cfg)

    # 4) build physics baseline on unscaled physical features
    if getattr(cfg, "use_physics_baseline", False):
        splits = build_physics_residuals(splits)
        for k in ("train", "val", "test"):
            splits[k] = shift_phys_targets(splits[k], cfg)

    # 5) NOW scale features
    scaler = fit_transformers(splits["train"], cfg)
    for k in ("train", "val", "test"):
        if len(splits[k]) > 0:
            splits[k] = apply_transformers(splits[k], scaler, cfg)

    # 6) scale target columns actually fed to model
    from .config import resolved_target_cols
    targ_cols = resolved_target_cols(cfg)

    t_scaler = fit_target_scaler(splits["train"], targ_cols)
    for k in ("train", "val", "test"):
        if len(splits[k]) > 0:
            splits[k] = apply_target_scaler(splits[k], targ_cols, t_scaler)

    splits["_target_scaler"] = t_scaler

    runs_dir = Path(".runs")
    runs_dir.mkdir(exist_ok=True)

    with open(runs_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(runs_dir / "target_scaler.pkl", "wb") as f:
        pickle.dump(t_scaler, f)

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
def make_features_for_scenario(df_raw):
    """
    EXACTLY mirrors the training preprocessing from prepare_frames(),
    except: no split, no scaling, no targets.
    """
    df = canonicalize_units_and_names(df_raw, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    return df
