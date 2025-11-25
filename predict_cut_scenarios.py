#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import canonicalize_units_and_names, downsample, add_regime_features
from src.models import TCNRegressor


# ------------------------------
# Load scalers (same as training)
# ------------------------------
def load_scalers():
    with open(".runs/feature_scaler.pkl", "rb") as f:
        f_scaler = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        t_scaler = pickle.load(f)
    return f_scaler, t_scaler


# ------------------------------
# Preprocess scenario like training
# ------------------------------
def preprocess_scenario(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_units_and_names(df, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    return df.reset_index(drop=True)


# ------------------------------
# Predict PM1/PM2.5/PM10 counts + SUM
# and return (pm_sum, time_axis)
# ------------------------------
def predict_pm_sum(csv_path: str, start_row: int, end_row: int):
    print(f"\n=== Predicting PM counts for {csv_path} ===")

    # 1) Load CSV and cut rows (end_row is EXCLUSIVE, so 82:428 means 82..427)
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.iloc[start_row:end_row].reset_index(drop=True)
    print("Cut rows (raw):", len(df_raw))

    # 2) Apply same feature engineering as training
    df_feat = preprocess_scenario(df_raw)
    print("Rows after downsample+features:", len(df_feat))

    feat_cols = resolved_feature_cols(cfg)   # 12 base features
    targ_cols = resolved_target_cols(cfg)    # ["PM1_count","PM2_5_count","PM10_count"]

    # 3) Build time axis (prefer physical time if present)
    if "time_s" in df_feat.columns:
        time = df_feat["time_s"].to_numpy()
    else:
        # fallback: synthetic time with target_hz
        time = np.arange(len(df_feat)) / cfg.target_hz

    # 4) Load scalers
    f_scaler, t_scaler = load_scalers()

    # 5) Feature scaling
    df_scaled = df_feat.copy()
    df_scaled[feat_cols] = f_scaler.transform(df_feat[feat_cols])

    # 6) Load trained TCN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCNRegressor(
        input_size=len(feat_cols),
        out_dim=len(targ_cols),
        channels_list=list(cfg.tcn_channels),
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.dropout,
    ).to(device)

    model.load_state_dict(torch.load(".runs/tcn_best.pt", map_location=device))
    model.eval()

    # 7) Predict frame-by-frame (T=1 for each step)
    preds_scaled = []
    with torch.no_grad():
        for i in range(len(df_scaled)):
            x = df_scaled[feat_cols].iloc[i].values.astype(np.float32)
            x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(device)  # (B=1, T=1, F)
            y = model(x)  # shape (1, 1, 3) or (1, 3) -> squeeze below
            preds_scaled.append(y.cpu().numpy().squeeze())

    preds_scaled = np.vstack(preds_scaled)

    # 8) Inverse-transform targets back to real PM counts
    preds_real = t_scaler.inverse_transform(preds_scaled)

    pm1_idx  = targ_cols.index("PM1_count")
    pm25_idx = targ_cols.index("PM2_5_count")
    pm10_idx = targ_cols.index("PM10_count")

    pm1_pred  = preds_real[:, pm1_idx]
    pm25_pred = preds_real[:, pm25_idx]
    pm10_pred = preds_real[:, pm10_idx]

    pm_sum_pred = pm1_pred + pm25_pred + pm10_pred

    # quick sanity print
    print("First 5 PM_SUM_pred:", pm_sum_pred[:5])

    # 9) Save per-scenario CSV
    Path("figures").mkdir(exist_ok=True)
    out_csv = f"figures/{Path(csv_path).stem}_pm_counts_predicted.csv"
    pd.DataFrame({
        "time_s": time,
        "PM1_pred": pm1_pred,
        "PM2_5_pred": pm25_pred,
        "PM10_pred": pm10_pred,
        "PM_SUM_pred": pm_sum_pred,
    }).to_csv(out_csv, index=False)
    print("Saved predictions:", out_csv)

    return pm_sum_pred, time


# ------------------------------
# RUN FOR BOTH SCENARIOS + one plot
# ------------------------------
if __name__ == "__main__":
    # Your specified cuts:
    # Booster_10_06_44 => lines 82:428
    # Stopper_10_14_44 => lines 143:489
    booster_sum, booster_t = predict_pm_sum("data/processed/booster_clean.csv",
                                            start_row=82, end_row=428)
    stopper_sum, stopper_t = predict_pm_sum("data/processed/stopper_clean.csv",
                                            start_row=143, end_row=489)

    # Align time axes to start at 0 for each scenario
    booster_t = booster_t - booster_t[0]
    stopper_t = stopper_t - stopper_t[0]

    # If lengths differ after preprocessing, trim to common min
    n = min(len(booster_sum), len(stopper_sum))
    booster_t = booster_t[:n]
    stopper_t = stopper_t[:n]
    booster_sum = booster_sum[:n]
    stopper_sum = stopper_sum[:n]

    # Combined time-based plot
    Path("figures").mkdir(exist_ok=True)
    out_png = "figures/booster_vs_stopper_pm_sum_over_time.png"

    plt.figure(figsize=(10, 5))
    plt.plot(booster_t, booster_sum, label="Booster PM_SUM_pred")
    plt.plot(stopper_t, stopper_sum, label="Stopper PM_SUM_pred")
    plt.xlabel("Time [s] (cut segment)")
    plt.ylabel("PM count (PM1 + PM2.5 + PM10)")
    plt.title("Booster vs Stopper — Predicted PM SUM over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print("Saved combined time plot:", out_png)
