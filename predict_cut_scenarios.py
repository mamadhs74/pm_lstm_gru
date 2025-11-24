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
        fsc = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        tsc = pickle.load(f)
    return fsc, tsc


# ------------------------------
# Preprocess scenario
# ------------------------------
def preprocess(df):
    df = canonicalize_units_and_names(df, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    return df.reset_index(drop=True)


# ------------------------------
# Build model
# ------------------------------
def load_model(feat_cols):
    model = TCNRegressor(
        input_size=len(feat_cols),
        out_dim=len(resolved_target_cols(cfg)),
        channels_list=list(cfg.tcn_channels),
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.dropout,
    )
    model.load_state_dict(torch.load(".runs/tcn_best.pt", map_location="cpu"))
    model.eval()
    return model


# ------------------------------
# Sliding-window prediction
# ------------------------------
def predict_timeseries(df_scaled, model, feat_cols):
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    X = df_scaled[feat_cols].values.astype(np.float32)

    preds = []
    for start in range(0, len(X) - window + 1, stride):
        xb = torch.tensor(X[start:start+window]).unsqueeze(0)
        with torch.no_grad():
            yp = model(xb).cpu().numpy().squeeze()
        preds.append(yp)

    return np.array(preds)


# ------------------------------
# Predict all 3 PM counts + summed total
# ------------------------------
def predict_all_counts(csv_path):
    df_raw = pd.read_csv(csv_path)
    df_feat = preprocess(df_raw)

    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    # Load scalers
    fsc, tsc = load_scalers()

    # Feature scaling
    df_scaled = df_feat.copy()
    df_scaled[feat_cols] = fsc.transform(df_feat[feat_cols])

    # Load model
    model = load_model(feat_cols)

    # Predict (scaled residual-shift)
    preds_scaled = predict_timeseries(df_scaled, model, feat_cols)
    preds_real = tsc.inverse_transform(preds_scaled)

    # Extract indices
    idx_pm1 = targ_cols.index("PM1_count_resid_shift")
    idx_pm25 = targ_cols.index("PM2_5_count_resid_shift")
    idx_pm10 = targ_cols.index("PM10_count_resid_shift")

    PM1 = preds_real[:, idx_pm1]
    PM25 = preds_real[:, idx_pm25]
    PM10 = preds_real[:, idx_pm10]

    # Total particle count
    PM_TOTAL = PM1 + PM25 + PM10

    return PM1, PM25, PM10, PM_TOTAL


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    booster_path = "data/processed/booster_clean.csv"
    stopper_path = "data/processed/stopper_clean.csv"

    # Predict both scenarios
    b_PM1, b_PM25, b_PM10, b_TOTAL = predict_all_counts(booster_path)
    s_PM1, s_PM25, s_PM10, s_TOTAL = predict_all_counts(stopper_path)

    # Save to CSV
    pd.DataFrame({
        "PM1_pred": b_PM1,
        "PM2_5_pred": b_PM25,
        "PM10_pred": b_PM10,
        "PM_TOTAL_pred": b_TOTAL
    }).to_csv("booster_count_predictions.csv", index=False)

    pd.DataFrame({
        "PM1_pred": s_PM1,
        "PM2_5_pred": s_PM25,
        "PM10_pred": s_PM10,
        "PM_TOTAL_pred": s_TOTAL
    }).to_csv("stopper_count_predictions.csv", index=False)

    # Plot total count comparison
    plt.figure(figsize=(12, 6))
    plt.plot(b_TOTAL, label="Booster PM_TOTAL")
    plt.plot(s_TOTAL, label="Stopper PM_TOTAL")
    plt.title("Predicted Total Particle Count (PN_total) — Booster vs Stopper")
    plt.xlabel("Window index (stride = 1s)")
    plt.ylabel("Total particle count")
    plt.legend()
    plt.grid()
    plt.savefig("figures/booster_vs_stopper_PM_TOTAL.png")
    plt.show()

    print("Saved predictions + plot.")
