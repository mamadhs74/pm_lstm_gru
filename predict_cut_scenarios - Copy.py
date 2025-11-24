#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import canonicalize_units_and_names, downsample, add_regime_features
from src.models import TCNRegressor
from src.visualize import plot_overlay


# ------------------------------
# Load scalers
# ------------------------------
def load_scalers():
    with open(".runs/feature_scaler.pkl", "rb") as f:
        f_scaler = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        t_scaler = pickle.load(f)
    return f_scaler, t_scaler


# ------------------------------
# Basic feature pipeline
# ------------------------------
def preprocess_scenario(df):
    df = canonicalize_units_and_names(df, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    return df


# ------------------------------
# PM10 prediction for a scenario
# ------------------------------
def predict_pm10(csv_path, start_row, end_row):
    print(f"\n=== Predicting PM10 for {csv_path} ===")

    # 1) Load CSV and cut rows
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.iloc[start_row:end_row].reset_index(drop=True)
    print("Cut rows:", len(df_raw))

    # 2) Apply feature engineering
    df_feat = preprocess_scenario(df_raw)

    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    # 3) Load scalers
    f_scaler, t_scaler = load_scalers()

    # 4) Apply feature scaling
    df_scaled = df_feat.copy()
    df_scaled[feat_cols] = f_scaler.transform(df_feat[feat_cols])

    # 5) Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCNRegressor(
        len(feat_cols),
        len(targ_cols),
        list(cfg.tcn_channels),
        cfg.tcn_kernel_size,
        cfg.dropout,
    ).to(device)

    model.load_state_dict(torch.load(".runs/tcn_best.pt", map_location=device))
    model.eval()

    # 6) Predict frame-by-frame (NO windows)
    preds_scaled = []

    with torch.no_grad():
        for i in range(len(df_scaled)):
            x = df_scaled[feat_cols].iloc[i].values.astype(np.float32)
            x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(device)  # shape (1,1,F)
            y = model(x)
            preds_scaled.append(y.cpu().numpy().squeeze())

    preds_scaled = np.vstack(preds_scaled)

    # 7) Inverse-transform
    preds_real = t_scaler.inverse_transform(preds_scaled)

    # 8) Extract PM10
    pm10_col = [c for c in targ_cols if c.startswith("PM10_")][0]
    pm10_idx = targ_cols.index(pm10_col)

    pm10_pred = preds_real[:, pm10_idx]

    # 9) Save CSV
    out_csv = f"figures/{Path(csv_path).stem}_pm10_predicted.csv"
    pd.DataFrame({"PM10_pred": pm10_pred}).to_csv(out_csv, index=False)
    print("Saved predictions:", out_csv)

    # 10) Save plot
    out_png = f"figures/{Path(csv_path).stem}_pm10_plot.png"
    df_plot = pd.DataFrame({"PM10_pred": pm10_pred})
    plot_overlay(df_plot, out_png, title=f"{Path(csv_path).stem} — PM10 predicted")
    print("Saved plot:", out_png)

    return pm10_pred


# ------------------------------
# RUN FOR BOTH SCENARIOS
# ------------------------------
if __name__ == "__main__":
    booster_pred = predict_pm10("data/processed/booster_clean.csv", 82, 428)
    stopper_pred = predict_pm10("data/processed/stopper_clean.csv", 143, 489)
