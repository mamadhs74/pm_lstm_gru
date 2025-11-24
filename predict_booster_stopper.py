import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import canonicalize_units_and_names, downsample, add_regime_features
from src.models import TCNRegressor

# ---------------------------------------------------
# Load scalers used during training
# ---------------------------------------------------
def load_scalers():
    with open(".runs/feature_scaler.pkl", "rb") as f:
        fsc = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        tsc = pickle.load(f)
    return fsc, tsc

# ---------------------------------------------------
# Preprocess scenario like training pipeline
# ---------------------------------------------------
def preprocess(df):
    df = canonicalize_units_and_names(df, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    return df.reset_index(drop=True)

# ---------------------------------------------------
# Build inference model
# ---------------------------------------------------
def load_model(feat_cols):
    n_in = len(feat_cols)
    n_out = len(resolved_target_cols(cfg))

    model = TCNRegressor(
        input_size=n_in,
        out_dim=n_out,
        channels_list=list(cfg.tcn_channels),
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.dropout,
    )
    model.load_state_dict(torch.load(".runs/tcn_best.pt", map_location="cpu"))
    model.eval()
    return model

# ---------------------------------------------------
# Predict full time series
# ---------------------------------------------------
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

# ---------------------------------------------------
# Predict PM10 (ug/m3)
# ---------------------------------------------------
def predict_pm10(csv_path, cut_start=None, cut_end=None):
    df_raw = pd.read_csv(csv_path)

    # ⭐ APPLY REQUIRED SCENARIO CUTTING HERE ⭐
    if cut_start is not None and cut_end is not None:
        df_raw = df_raw.iloc[cut_start:cut_end].reset_index(drop=True)
        print(f"Applied cut: rows {cut_start}:{cut_end} → new length {len(df_raw)}")

    df_feat = preprocess(df_raw)

    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    fsc, tsc = load_scalers()

    df_scaled = df_feat.copy()
    df_scaled[feat_cols] = fsc.transform(df_feat[feat_cols])

    model = load_model(feat_cols)

    preds_scaled = predict_timeseries(df_scaled, model, feat_cols)
    preds_real = tsc.inverse_transform(preds_scaled)

    idx = targ_cols.index("PM10_ug_per_m3_resid_shift")
    pm10 = preds_real[:, idx]

    return pm10

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":

    # ⭐ Apply required slicing from Marco ⭐
    booster_pm10 = predict_pm10(
        "data/processed/booster_clean.csv",
        cut_start=82, cut_end=428
    )

    stopper_pm10 = predict_pm10(
        "data/processed/stopper_clean.csv",
        cut_start=143, cut_end=489
    )

    # Save predictions
    pd.DataFrame({"PM10_pred": booster_pm10}).to_csv(
        "booster_predictions_pm10.csv", index=False)
    pd.DataFrame({"PM10_pred": stopper_pm10}).to_csv(
        "stopper_predictions_pm10.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(booster_pm10, label="Booster PM10 Prediction")
    plt.plot(stopper_pm10, label="Stopper PM10 Prediction")
    plt.title("Predicted PM10 Over Time — Booster vs Stopper")
    plt.xlabel("Window Index (5 Hz, stride=1s)")
    plt.ylabel("PM10 [µg/m³]")
    plt.legend()
    plt.grid()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/booster_vs_stopper_pm10.png", dpi=300)

    print("\nSaved: figures/booster_vs_stopper_pm10.png\n")
