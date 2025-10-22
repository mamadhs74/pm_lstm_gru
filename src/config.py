# src/config.py (drop-in if you want my tuned defaults)
from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = "data/processed/1_clean.csv"
    processed_dir: str = "data/processed"
    seed: int = 42

    # sampling & windowing
    orig_hz: int = 10
    target_hz: int = 5
    window_s: int = 40           # 200 steps
    target_shift_s: int = 20     # predict t+20s
    stride_steps: int = 5        # 1 s stride at 5 Hz

    # features/targets (clean names)
    feature_cols = [
        "speed_kmh","acc_long_ms2","brake_bar","yaw_deg_s",
        "acc_lat_g","steer_rate_deg_s","rpm","ambient_temp_c",
        "wheel_fl_kmh","wheel_fr_kmh","wheel_rl_kmh","wheel_rr_kmh",
    ]
    target_cols = ["PM10_ug_per_m3","PM2_5_ug_per_m3","PM1_ug_per_m3"]

    # model
    model_type: str = "gru"     # lstm or "gru"
    hidden_size: int = 128
    num_layers: int = 2          # dropout only active if >1
    dropout: float = 0.2
    conv_frontend: bool = True
    conv_channels: int = 64

    # training
    batch_size: int = 256
    max_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # loss / weights
    use_huber: bool = True
    huber_delta: float = 1.0
    regime_weights = {"idle": 0.7, "brake": 2.0, "cruise": 1.0}

    # CV / splitting
    n_folds: int = 5
    purge_seconds: int = 60

cfg = Config()
