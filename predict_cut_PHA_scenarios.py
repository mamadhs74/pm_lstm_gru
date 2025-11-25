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
# Load scalers
# ------------------------------
def load_scalers():
    with open(".runs/feature_scaler.pkl", "rb") as f:
        f_scaler = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        t_scaler = pickle.load(f)
    return f_scaler, t_scaler


# ------------------------------
# Preprocess like training
# ------------------------------
def preprocess_scenario(df):
    df = canonicalize_units_and_names(df, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    return df.reset_index(drop=True)


# ------------------------------
# Predict P_HA over time
# ------------------------------
def predict_PHA(csv_path, start, end):
    print(f"\n=== PREDICTING P_HA for {csv_path} ===")

    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.iloc[start:end].reset_index(drop=True)
    print("Cut rows:", len(df_raw))

    df_feat = preprocess_scenario(df_raw)

    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)  # ["P_HA"]

    # construct time axis (real or synthetic)
    if "time_s" in df_feat.columns:
        time = df_feat["time_s"].to_numpy()
        time = time - time[0]   # start at 0
    else:
        time = np.arange(len(df_feat)) / cfg.target_hz

    # Load scalers
    fsc, tsc = load_scalers()

    df_scaled = df_feat.copy()
    df_scaled[feat_cols] = fsc.transform(df_feat[feat_cols])

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCNRegressor(
        input_size=len(feat_cols),
        out_dim=len(targ_cols),   # = 1
        channels_list=list(cfg.tcn_channels),
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.dropout
    ).to(device)

    model.load_state_dict(torch.load(".runs/tcn_best.pt", map_location=device))
    model.eval()

    # Frame-by-frame predictions
    preds_scaled = []
    with torch.no_grad():
        for i in range(len(df_scaled)):
            x = torch.tensor(df_scaled[feat_cols].iloc[i].values,
                             dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            y = model(x)
            preds_scaled.append(y.cpu().numpy().squeeze())

    preds_scaled = np.vstack(preds_scaled)

    # Inverse transform to real P_HA units
    preds_real = tsc.inverse_transform(preds_scaled).reshape(-1)

    # Save
    out_csv = f"figures/{Path(csv_path).stem}_PHA_pred.csv"
    pd.DataFrame({"time_s": time, "P_HA_pred": preds_real}).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    return time, preds_real


# ------------------------------
# Run both scenarios + combined plot
# ------------------------------
if __name__ == "__main__":
    booster_t, booster_pred = predict_PHA("data/processed/booster_clean.csv", 82, 428)
    stopper_t, stopper_pred = predict_PHA("data/processed/stopper_clean.csv", 143, 489)

    # align lengths
    n = min(len(booster_pred), len(stopper_pred))
    booster_t, booster_pred = booster_t[:n], booster_pred[:n]
    stopper_t, stopper_pred = stopper_t[:n], stopper_pred[:n]

    # Plot both
    plt.figure(figsize=(10,5))
    plt.plot(booster_t, booster_pred, label="Booster — P_HA Prediction")
    plt.plot(stopper_t, stopper_pred, label="Stopper — P_HA Prediction")
    plt.xlabel("Time [s]")
    plt.ylabel("P_HA")
    plt.title("Predicted P_HA Over Time — Booster vs Stopper")
    plt.grid(True)
    plt.legend()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/booster_vs_stopper_PHA_over_time.png")
    plt.close()

    print("Saved combined plot: figures/booster_vs_stopper_PHA_over_time.png")
