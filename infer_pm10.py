import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import canonicalize_units_and_names, downsample
from src.models import TCNRegressor, RNNRegressor


# -----------------------------
# Load saved scalers
# -----------------------------
def load_scalers():
    with open(".runs/feature_scaler.pkl", "rb") as f:
        f_scaler = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        t_scaler = pickle.load(f)
    return f_scaler, t_scaler


# -----------------------------
# Only apply SAME training features
# -----------------------------
def scenario_features(df_raw):
    df = canonicalize_units_and_names(df_raw, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    return df


# -----------------------------
# Build model
# -----------------------------
def build_model(device, feat_cols):

    targs = resolved_target_cols(cfg)

    if cfg.model_type.lower() in ("lstm", "gru"):
        return RNNRegressor(
            input_size=len(feat_cols),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.model_type,
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=len(targs),
        ).to(device)

    # TCN
    return TCNRegressor(
        len(feat_cols),
        len(targs),
        list(cfg.tcn_channels),
        cfg.tcn_kernel_size,
        cfg.dropout,
    ).to(device)


# -----------------------------
# Run inference on a scenario
# -----------------------------
def predict_pm10(csv_path, ckpt_path, tag):
    print(f"\n=== Predicting PM10 for {csv_path} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_raw = pd.read_csv(csv_path)
    df_feat = scenario_features(df_raw)

    feat_cols = resolved_feature_cols(cfg)   # EXACT 12 training features
    targ_cols = resolved_target_cols(cfg)

    # Ensure alignment
    missing = [c for c in feat_cols if c not in df_feat.columns]
    if missing:
        print("Missing:", missing)
        return

    # Load scalers
    f_scaler, t_scaler = load_scalers()

    # Scale features
    X = f_scaler.transform(df_feat[feat_cols])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, F)

    # Load model
    model = build_model(device, feat_cols)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Predict all windows at once
    with torch.no_grad():
        pred_scaled = model(X).cpu().numpy()

    # reverse scaling
    pred_real = t_scaler.inverse_transform(pred_scaled)

    # find PM10 column
    pm10_col = [c for c in targ_cols if c.startswith("PM10")][0]
    idx_pm10 = targ_cols.index(pm10_col)

    pm10_pred = pred_real[:, idx_pm10]

    # save
    out_df = pd.DataFrame({"PM10_pred": pm10_pred})
    out_path = f"figures/{Path(csv_path).stem}_{tag}_pm10_timeseries.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Saved PM10 prediction: {out_path}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    ckpt = ".runs/tcn_best.pt"

    predict_pm10("data/processed/booster_clean.csv",  ckpt, "plain")
    predict_pm10("data/processed/stopper_clean.csv",   ckpt, "plain")
