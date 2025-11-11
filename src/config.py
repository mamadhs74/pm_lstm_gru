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
    stride_steps: int = 10        # 1 s stride @5 Hz

    # ---- feature/target columns (clean CSV names) ----
    feature_cols = [
        "speed_kmh","acc_long_ms2","brake_bar","yaw_deg_s",
         "acc_lat_ms2","steer_rate_deg_s","rpm","ambient_temp_c",
        "wheel_fl_kmh","wheel_fr_kmh","wheel_rl_kmh","wheel_rr_kmh",
    ]
    target_cols = ["PM10_ug_per_m3","PM2_5_ug_per_m3","PM1_ug_per_m3"]

        # ---- model ----
    model_type: str = "tcn"  # "lstm", "gru", or "tcn"
    hidden_size: int = 128       # used by RNN
    num_layers: int = 2          # used by RNN
    dropout: float = 0.4
    conv_frontend: bool = True   # used by RNN
    conv_channels: int = 64      # used by RNN

    # ---- TCN-specific ----
    tcn_channels = (64,64,64,64,64,64)  # per residual block (dilations 1,2,4,…)
    tcn_kernel_size: int = 3


    # ---- training ----
    # training
    batch_size: int = 256
    max_epochs: int = 80            
    early_stopping_patience: int = 10
    lr: float = 5e-4  
    weight_decay: float = 1e-4  

    # loss / weights
    use_huber: bool = True
    huber_delta: float = 1.0
    regime_weights = {"idle": 1.2, "brake": 1.8, "cruise": 1.0}  # (optional tweak)

    # ---- CV / splitting ----
    n_folds: int = 5
    purge_seconds: int = 180

    # ---- column names (time/speed/alt/dist) ----
    time_col: str   = "time_s"
    speed_col: str  = "speed_kmh"   # km/h in your CSV
    alt_col: str    = "altitude_m"  # optional, for grade
    dist_col: str   = "distance_m"  # optional, for grade fallback

    # ===== Vehicle geometry & measured constants =====
    # gravity (your value)
    g: float = 9.80665

    # wheelbase (m)
    wheelbase_m: float = 3.0
    h_cg_m: float         = 0.62     
    # track widths (m). Use midpoints of your ranges (spacers handled in those ranges already).
    # front: 1622–1634 mm → ~1.628 m
    # rear : 1628–1640 mm → ~1.634 m
    track_front_m: float = (1622.0 + 1634.0) / 2000.0
    track_rear_m:  float = (1628.0 + 1640.0) / 2000.0

    # tire radius from 255/45 R18 (mm → m): 18" rim (457.2 mm) + 2*114.75 mm sidewalls → radius ≈ 0.343 m
    wheel_radius_m: float = (457.2 + 2 * 114.75) / 2000.0

    # PM probe height (m)
    probe_height_m: float = 0.13

    # corner weights (kg) you measured
    weight_fl_kg: float = 681.0
    weight_fr_kg: float = 630.0
    weight_rl_kg: float = 464.5
    weight_rr_kg: float = 484.0

    # total mass from corner weights (kg)
    vehicle_mass_kg: float = weight_fl_kg + weight_fr_kg + weight_rl_kg + weight_rr_kg  # ≈ 2259.5

    # longitudinal CG from front axle (m): (rear mass / total) * wheelbase
    cg_longitudinal_m: float = ((weight_rl_kg + weight_rr_kg) / vehicle_mass_kg) * wheelbase_m  # ≈ 1.26 m

    # vertical CG height estimate (m). Increase to ~0.65–0.70 when heavily loaded.
    cg_height_m: float = 0.62

    # braking distribution baseline (front share)
    brake_beta_front: float = 0.60

    # effective brake radius (m) (caliper/disc)
    brake_reff_m: float = 0.15

    # environment / drag
    rho_air: float = 1.225
    CdA: float     = 0.65
    Crr: float     = 0.012

    # ===== physics-baseline options =====
    physics_ma_window: int = 5
    grade_smooth_window: int = 9
    use_physics_baseline: bool = False  # train on (PM - PF * W_spec)

    # (optional) dynamic front share using simple longitudinal weight transfer
    use_dynamic_brake_share: bool = True   # set True to enable
    beta_clip_min: float = 0.50
    beta_clip_max: float = 0.90


def resolved_feature_cols(cfg: Config):
    return cfg.feature_cols + ["idle", "brake_event", "dv", "jerk"]

def resolved_target_cols(cfg: Config):
    if getattr(cfg, "use_physics_baseline", False):
        return [f"{c}_resid_phys_shift" for c in cfg.target_cols]
    else:
        return [f"{c}_resid_shift" for c in cfg.target_cols]

cfg = Config()
