# src/physics.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _ma(x: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(x).rolling(k, min_periods=1, center=True).mean().to_numpy()

def _grad(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(y, x, edge_order=1)

def compute_grade_from_alt(time_s, alt_m, dist_m=None, smooth=9):
    a = _ma(np.asarray(alt_m, float), smooth)
    t = np.asarray(time_s, float)
    if dist_m is not None and np.any(np.diff(dist_m) != 0):
        d = _ma(np.asarray(dist_m, float), smooth)
        sdot = _grad(d, t) + 1e-9  # path speed (m/s)
    else:
        sdot = np.ones_like(t)
    adot = _grad(a, t)
    return _ma(np.arctan2(adot, sdot), smooth)

def compute_friction_work_from_df(df: pd.DataFrame, cfg) -> Tuple[np.ndarray, np.ndarray]:
    t = df[cfg.time_col].to_numpy(float)
    v = df[cfg.speed_col].to_numpy(float) * (1000.0 / 3600.0)  # km/h → m/s
    v_s = _ma(v, cfg.physics_ma_window)
    dt = np.diff(t, prepend=t[0])
    ax = _grad(v_s, t)

    # grade (radians), optional
    if cfg.alt_col in df.columns:
        alt = df[cfg.alt_col].to_numpy(float)
        dist = df[cfg.dist_col].to_numpy(float) if cfg.dist_col in df.columns else None
        grade = compute_grade_from_alt(t, alt, dist, cfg.grade_smooth_window)
    else:
        grade = np.zeros_like(v_s)

    m, g = float(cfg.vehicle_mass_kg), 9.81
    rho, CdA, Crr = float(cfg.rho_air), float(cfg.CdA), float(cfg.Crr)
    reff, beta = float(cfg.brake_reff_m), float(cfg.brake_beta_front)

    # Forces
    Fr = Crr * m * g * np.cos(grade)     # rolling
    Fa = 0.5 * rho * CdA * (v_s**2)      # aero
    Fg = m * g * np.sin(grade)           # grade
    F_long = m * ax                      # inertia

    # Brake force only when needed (decel after resistances)
    F_brake = np.maximum(0.0, -(F_long + Fa + Fr + Fg))
    T_brake_front = beta * F_brake * reff
    P_brake = T_brake_front * (v_s / reff)       # W
    W_fric = np.cumsum(P_brake * dt)             # J
    W_spec = W_fric / m                          # J/kg
    return W_fric, W_spec

def fit_pf_linear(work: np.ndarray, pm: np.ndarray) -> float:
    w = work.astype(float); y = pm.astype(float)
    return float(np.dot(w, y) / (np.dot(w, w) + 1e-12))

def fit_pf_per_target(train_df: pd.DataFrame, cfg) -> Dict[str, float]:
    _, Wspec = compute_friction_work_from_df(train_df, cfg)
    pf = {}
    for col in cfg.target_cols:
        short = col.split("_ug")[0]
        pf[short] = fit_pf_linear(Wspec, train_df[col].to_numpy(float))
    return pf

def apply_physics_baseline(df: pd.DataFrame, cfg, pf: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    _, Wspec = compute_friction_work_from_df(out, cfg)
    out["W_spec"] = Wspec
    for col in cfg.target_cols:
        short = col.split("_ug")[0]
        yphys = pf[short] * Wspec
        out[f"{short}_phys"] = yphys
        out[f"{col}_resid_phys"] = out[col].to_numpy(float) - yphys
    return out
