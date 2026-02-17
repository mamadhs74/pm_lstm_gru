#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import canonicalize_units_and_names, downsample, add_regime_features
from src.models import TCNRegressor


# ---------------------------------------
# Load scalers
# ---------------------------------------
def load_scalers():
    with open(".runs/feature_scaler.pkl", "rb") as f:
        fsc = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        tsc = pickle.load(f)
    return fsc, tsc


# ---------------------------------------
# Preprocess
# ---------------------------------------
def preprocess(df):
    df = canonicalize_units_and_names(df, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    return df.reset_index(drop=True)


# ---------------------------------------
# Predict using window-based TCN
# ---------------------------------------
def predict_series(df_scaled, model, feat_cols):
    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps

    X = df_scaled[feat_cols].values.astype(np.float32)
    preds = []

    for start in range(0, len(X) - window + 1, stride):
        xb = torch.tensor(X[start:start+window]).unsqueeze(0)
        with torch.no_grad():
            yp = model(xb).cpu().numpy().squeeze()
        preds.append(yp)

    return np.array(preds)


# ---------------------------------------
# Plot truth vs predicted P_VA
# ---------------------------------------
def plot_pva(csv_path):

    # Load CSV
    df = pd.read_csv(csv_path)

    # Preprocess
    df_feat = preprocess(df)
    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    # Load scalers
    fsc, tsc = load_scalers()

    df_scaled = df_feat.copy()
    df_scaled[feat_cols] = fsc.transform(df_feat[feat_cols])

    # Load model
    model = TCNRegressor(
        input_size=len(feat_cols),
        out_dim=len(targ_cols),
        channels_list=list(cfg.tcn_channels),
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.dropout,
    )
    model.load_state_dict(torch.load(".runs/tcn_best.pt", map_location="cpu"))
    model.eval()

    # Predict
    preds_scaled = predict_series(df_scaled, model, feat_cols)

    # FIX 1 — remove NaNs
    preds_scaled = preds_scaled[~np.isnan(preds_scaled)]

    # FIX 2 — reshape to 2D
    preds_real = tsc.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    # Target column
    pva_col = targ_cols[0]

    # FIX 3 — align truth length with predictions
    pva_true = df_feat[pva_col].iloc[:len(preds_real)]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pva_true.values, label="True P_VA", color="blue")
    plt.plot(preds_real, label="Predicted P_VA", color="orange")
    plt.title("P_VA Over Time — Truth vs Prediction")
    plt.xlabel("Time Index")
    plt.ylabel("P_VA")
    plt.legend()
    plt.grid()

    out = "figures/P_VA_truth_vs_prediction.png"
    plt.savefig(out, dpi=300)
    plt.show()

    print("Saved:", out)


if __name__ == "__main__":
    plot_pva("data/processed/1_clean_decimal_fixed.csv")
