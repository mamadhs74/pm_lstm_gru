# src/config.py
from dataclasses import dataclass
import math

@dataclass
class Config:
    # ---- data paths ----
    data_path: str = "data/processed/1_clean.csv"
    processed_dir: str = "data/processed"
    seed: int = 42

    # ---- sampling & windowing ----
    orig_hz: int = 10
    target_hz: int = 5
    window_s: int = 40           # 200 steps @5 Hz
    target_shift_s: int = 20     # predict t+20 s
    stride_steps: int = 5        # 1 s stride @5 Hz

    # ---- feature/target columns (clean CSV names) ----
    feature_cols = [
        "speed_kmh","acc_long_ms2","brake_bar","yaw_deg_s",
        "acc_lat_g","steer_rate_deg_s","rpm","ambient_temp_c",
        "wheel_fl_kmh","wheel_fr_kmh","wheel_rl_kmh","wheel_rr_kmh",
    ]
    target_cols = ["PM10_ug_per_m3","PM2_5_ug_per_m3","PM1_ug_per_m3"]

    # ---- model (choose "lstm" or "gru") ----
    model_type: str = "lstm"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    conv_frontend: bool = True
    conv_channels: int = 64

    # ---- training ----
    # training
    batch_size: int = 256
    max_epochs: int = 80            # keep high; early stopping will cut it
    early_stopping_patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # loss / weights
    use_huber: bool = True
    huber_delta: float = 1.0
    regime_weights = {"idle": 1.2, "brake": 1.8, "cruise": 1.0}  # (optional tweak)

    # ---- CV / splitting ----
    n_folds: int = 5
    purge_seconds: int = 60

    # ---- column names (time/speed/alt/dist) ----
    time_col: str   = "time_s"
    speed_col: str  = "speed_kmh"   # km/h in your CSV
    alt_col: str    = "altitude_m"  # optional, for grade
    dist_col: str   = "distance_m"  # optional, for grade fallback

    # ---- your measured vehicle geometry (used for future extensions / load transfer) ----
    # Radstand (wheelbase) = 3000 mm
    wheelbase_m: float = 3.000

    # Spurweite (track width). You provided ranges + 30 mm spacers per side.
    # We'll store approximate *final* tracks (mm → m). Adjust if you have exact final values.
    track_front_m: float = (1_628 + 0.060)  / 1000.0  # ~1628 mm + 60 mm total spacer ≈ 1.688 m
    track_rear_m:  float = (1_634 + 0.060)  / 1000.0  # ~1634 mm + 60 mm total spacer ≈ 1.694 m

    # Reifenradius from 255/45 R18:
    # 18" rim = 457.2 mm; sidewall = 0.45 * 255 = 114.75 mm → diameter ≈ 457.2 + 2*114.75 = 686.7 mm
    # radius ≈ 343.35 mm = 0.343 m
    wheel_radius_m: float = 0.343

    # Bremskraftverteilung (front share) unknown → keep typical value
    brake_beta_front: float = 0.6  # tune if you measure it

    # ---- physics / environment constants (edit if you know better) ----
    vehicle_mass_kg: float = 1600.0
    brake_reff_m: float    = 0.15     # effective brake radius (caliper/disc)
    rho_air: float         = 1.225
    CdA: float             = 0.65     # drag coeff * area
    Crr: float             = 0.012    # rolling resistance

    # ---- physics-baseline options ----
    physics_ma_window: int = 5
    grade_smooth_window: int = 9
    use_physics_baseline: bool = True  # trains on (PM - PF * W_spec)

def resolved_feature_cols(cfg: Config):
    return cfg.feature_cols + ["idle", "brake_event", "dv", "jerk"]

def resolved_target_cols(cfg: Config):
    if getattr(cfg, "use_physics_baseline", False):
        return [f"{c}_resid_phys_shift" for c in cfg.target_cols]
    else:
        return [f"{c}_resid_shift" for c in cfg.target_cols]

cfg = Config()
