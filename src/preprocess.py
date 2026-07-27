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
    for k in ("train", "val", "test", "stationary_test"):
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



def split_active_and_stationary_tail(
    df: pd.DataFrame,
    cfg,
):
    """
    Separate the terminal stationary regime using vehicle signals only.

    The detector does not use P_VA and therefore cannot select the test
    period according to target magnitude.
    """
    required = [
        "speed_kmh",
        "acc_long_ms2",
        "acc_lat_ms2",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Cannot identify stationary tail; missing columns: {missing}"
        )

    speed = pd.to_numeric(df["speed_kmh"], errors="coerce")
    acc_long = pd.to_numeric(
        df["acc_long_ms2"],
        errors="coerce",
    )
    acc_lat = pd.to_numeric(
        df["acc_lat_ms2"],
        errors="coerce",
    )

    stationary = (
        speed.abs().lt(
            float(cfg.stationary_speed_threshold_kmh)
        )
        & acc_long.abs().lt(
            float(cfg.stationary_acc_threshold_ms2)
        )
        & acc_lat.abs().lt(
            float(cfg.stationary_acc_threshold_ms2)
        )
    ).fillna(False)

    minimum_samples = int(
        cfg.stationary_min_duration_s * cfg.target_hz
    )

    if minimum_samples < 1:
        raise RuntimeError("stationary_min_duration_s is invalid")

    rolling_fraction = (
        stationary.astype(float)
        .rolling(
            minimum_samples,
            min_periods=minimum_samples,
        )
        .mean()
    )

    minimum_search_start = int(0.50 * len(df))
    candidates = []

    for end_index in np.flatnonzero(
        (rolling_fraction >= 0.95).to_numpy()
    ):
        start_index = int(end_index - minimum_samples + 1)

        if start_index < minimum_search_start:
            continue

        remaining_fraction = float(
            stationary.iloc[start_index:].mean()
        )

        if remaining_fraction >= 0.90:
            candidates.append(start_index)

    if not candidates:
        raise RuntimeError(
            "No terminal stationary block was found using the "
            "predefined vehicle-signal rule."
        )

    cutoff = int(min(candidates))

    active = df.iloc[:cutoff].copy().reset_index(drop=True)
    stationary_tail = (
        df.iloc[cutoff:]
        .copy()
        .reset_index(drop=True)
    )

    if len(active) < 1000:
        raise RuntimeError(
            f"Active block is unexpectedly short: {len(active)} rows"
        )

    if len(stationary_tail) < minimum_samples:
        raise RuntimeError(
            "Detected stationary block is shorter than required."
        )

    manifest = {
        "full_rows_5hz": int(len(df)),
        "active_rows_5hz": int(len(active)),
        "stationary_rows_5hz": int(len(stationary_tail)),
        "stationary_start_row_5hz": cutoff,
        "stationary_tail_fraction": float(
            stationary.iloc[cutoff:].mean()
        ),
        "stationary_rule": {
            "speed_abs_less_than_kmh": float(
                cfg.stationary_speed_threshold_kmh
            ),
            "acc_long_abs_less_than_ms2": float(
                cfg.stationary_acc_threshold_ms2
            ),
            "acc_lat_abs_less_than_ms2": float(
                cfg.stationary_acc_threshold_ms2
            ),
            "minimum_duration_s": int(
                cfg.stationary_min_duration_s
            ),
            "rolling_required_fraction": 0.95,
            "tail_required_fraction": 0.90,
        },
    }

    for time_col in [
        "Zeitstempel",
        "Time_since_midnight_UTC",
        "time_s",
    ]:
        if time_col in df.columns:
            manifest["stationary_start_time"] = str(
                df.iloc[cutoff][time_col]
            )
            manifest["time_column"] = time_col
            break

    return active, stationary_tail, manifest




def leak_free_split(
    df: pd.DataFrame,
    cfg,
) -> Dict[str, pd.DataFrame]:
    n = len(df)
    t0 = int(0.60 * n)
    t1 = int(0.80 * n)

    purge_seconds = int(
        getattr(
            cfg,
            "evaluation_purge_seconds",
            cfg.purge_seconds,
        )
    )
    purge = int(purge_seconds * cfg.target_hz)

    train_end = max(t0 - purge, 0)
    val_start = min(t0 + purge, n)
    val_end = max(t1 - purge, 0)
    test_start = min(t1 + purge, n)

    train = df.iloc[:train_end]
    val = (
        df.iloc[val_start:val_end]
        if val_end > val_start
        else df.iloc[0:0]
    )
    test = df.iloc[test_start:]

    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    minimum_required = int(
        cfg.window_s * cfg.target_hz
        + cfg.target_shift_s * cfg.target_hz
    )

    for name, part in splits.items():
        if len(part) <= minimum_required:
            raise RuntimeError(
                f"{name} split is too short: {len(part)} rows"
            )

    return splits

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
    T = 1
    alpha = dt / (T + dt)
    out = df.copy()
    for col in cfg.target_cols:
        ema_col = f"{col}_ema120"
        resid_col = f"{col}_resid"
        out[ema_col] = out[col].ewm(alpha=alpha, adjust=False).mean()
        out[resid_col] = (out[col] - out[ema_col]) / (np.abs(out[ema_col]) + 1e-6)
    return out



def shift_ema_targets(
    df: pd.DataFrame,
    cfg,
):
    shift_steps = int(
        cfg.target_shift_s * cfg.target_hz
    )
    out = df.copy()

    for col in cfg.target_cols:
        ema_col = f"{col}_ema120"
        resid_col = f"{col}_resid"

        future_target = out[col].shift(-shift_steps)
        causal_ema = out[ema_col]

        out[f"{col}_shift"] = future_target

        # Permanent physical copy. Direct target scaling must not
        # destroy the physical ground truth needed for evaluation.
        out[f"{col}_future_physical"] = future_target

        # Historical name retained for compatibility. This stores
        # EMA(t), not EMA(t+h).
        out[f"{ema_col}_shift"] = causal_ema

        out[f"{resid_col}_shift"] = (
            (future_target - causal_ema)
            / (np.abs(causal_ema) + 1e-6)
        )

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
    


def prepare_frames(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    # 1. Canonical physical columns
    df = canonicalize_units_and_names(df, cfg)

    # 2. Resample and generate causal dynamics features
    df = downsample(
        df,
        cfg.orig_hz,
        cfg.target_hz,
    )
    df = add_regime_features(df)

    # Preserve physical dynamics values before feature scaling.
    for col in [
        "speed_kmh",
        "acc_long_ms2",
        "acc_lat_ms2",
        "steer_rate_deg_s",
    ]:
        if col in df.columns:
            df[f"{col}_physical"] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # 3. Separate active operation and terminal stationary tail
    active_df, stationary_df, split_manifest = (
        split_active_and_stationary_tail(df, cfg)
    )

    # 4. Chronological active-driving split with temporal gaps
    splits = leak_free_split(active_df, cfg)
    splits["stationary_test"] = stationary_df

    frame_keys = (
        "train",
        "val",
        "test",
        "stationary_test",
    )

    # 5. Strictly causal one-second-ahead targets
    for key in frame_keys:
        splits[key] = build_residual_targets(
            splits[key],
            cfg,
        )
        splits[key] = shift_ema_targets(
            splits[key],
            cfg,
        )

    # 6. Optional physics residual
    if getattr(cfg, "use_physics_baseline", False):
        splits = build_physics_residuals(splits)
        for key in frame_keys:
            splits[key] = shift_phys_targets(
                splits[key],
                cfg,
            )

    # 7. Feature scaler fitted on active training data only
    scaler = fit_transformers(
        splits["train"],
        cfg,
    )

    for key in frame_keys:
        if len(splits[key]) > 0:
            splits[key] = apply_transformers(
                splits[key],
                scaler,
                cfg,
            )

    # 8. Target scaler fitted on active training data only
    from .config import resolved_target_cols

    target_cols = resolved_target_cols(cfg)
    target_scaler = fit_target_scaler(
        splits["train"],
        target_cols,
    )

    for key in frame_keys:
        if len(splits[key]) > 0:
            splits[key] = apply_target_scaler(
                splits[key],
                target_cols,
                target_scaler,
            )

    splits["_target_scaler"] = target_scaler
    splits["_split_manifest"] = split_manifest

    runs_dir = Path(".runs")
    runs_dir.mkdir(exist_ok=True)

    run_tag = (
        f"{cfg.target_mode}_seed{cfg.seed}"
    )

    for filename, obj in [
        ("feature_scaler.pkl", scaler),
        ("target_scaler.pkl", target_scaler),
        (f"feature_scaler_{run_tag}.pkl", scaler),
        (f"target_scaler_{run_tag}.pkl", target_scaler),
    ]:
        with open(runs_dir / filename, "wb") as handle:
            pickle.dump(obj, handle)

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
