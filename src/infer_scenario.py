# src/infer_scenario.py

import pandas as pd
import torch
import pickle
from pathlib import Path

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.datasets import WindowedDataset
from src.models import TCNRegressor, RNNRegressor
from src.visualize import plot_overlay


def load_scalers():
    """Load feature & target scalers saved during training."""
    with open(".runs/feature_scaler.pkl", "rb") as f:
        f_scaler = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        t_scaler = pickle.load(f)
    return f_scaler, t_scaler


def build_model(device):
    feats = resolved_feature_cols(cfg)
    targs = resolved_target_cols(cfg)
    mt = cfg.model_type.lower()

    if mt in ("lstm", "gru"):
        model = RNNRegressor(
            input_size=len(feats),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.model_type,
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=len(targs),
        )
    else:
        model = TCNRegressor(
            num_inputs=len(feats),
            num_channels=list(cfg.tcn_channels),
            kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            out_dim=len(targs),
        )

    return model.to(device)


def predict_scenario(csv_path, ckpt_path, tag):
    print(f"\n=== Scenario inference for {csv_path} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1) Load data ---
    df = pd.read_csv(csv_path)
    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    # --- 2) Load scalers from training ---
    f_scaler, t_scaler = load_scalers()

    # --- 3) Apply feature scaling (NO refit!!) ---
    df_scaled = df.copy()
    df_scaled[feat_cols] = f_scaler.transform(df[feat_cols])

    # --- 4) Build windows exactly like training (official approach) ---
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)

    ds = WindowedDataset(df_scaled, feat_cols, targ_cols, window, stride)

    # --- 5) Load trained model ---
    model = build_model(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- 6) Run inference ---
    preds = []
    with torch.no_grad():
        for xb, _ in ds:
            xb = xb.to(device)
            out = model(xb.unsqueeze(0)).cpu().numpy().squeeze()
            preds.append(out)

    preds = pd.DataFrame(preds, columns=[t + "_scaled_pred" for t in targ_cols])

    # --- 7) Inverse-transform predictions ---
    preds[targ_cols] = t_scaler.inverse_transform(preds[[t + "_scaled_pred" for t in targ_cols]])

    # --- 8) Trim DF for alignment ---
    df_plot = pd.DataFrame({
        "PM10_true": df["PM10_ug_per_m3"].iloc[window - 1:].reset_index(drop=True),
        "PM10_pred": preds["PM10_ug_per_m3"].reset_index(drop=True),
    })

    # --- 9) Plot ---
    out_path = f"figures/{Path(csv_path).stem}_{tag}_pm10_overlay.png"
    plot_overlay(df_plot, out_path,
                 title=f"{Path(csv_path).stem} – PM10 true vs predicted ({tag})")

    print("Saved:", out_path)


if __name__ == "__main__":
    booster = "data/processed/booster_clean.csv"
    stopper = "data/processed/stopper_clean.csv"

    ckpt_plain = ".runs/tcn_best.pt"     # or tcn_plain_best.pt
    ckpt_phys  = ".runs/tcn_phys_best.pt"

    predict_scenario(booster, ckpt_plain, "plain")
    predict_scenario(stopper, ckpt_plain, "plain")

    # If you have a physics model:
    # predict_scenario(booster, ckpt_phys, "phys")
    # predict_scenario(stopper, ckpt_phys, "phys")
