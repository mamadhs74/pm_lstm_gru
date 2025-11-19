# src/infer_scenario.py

import pandas as pd
import torch
import pickle
from pathlib import Path

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.datasets import WindowedDataset
from src.models import TCNRegressor, RNNRegressor
from src.visualize import plot_overlay
from src.preprocess import make_features_for_scenario


def load_scalers():
    """Load feature & target scalers saved during training."""
    with open(".runs/feature_scaler.pkl", "rb") as f:
        f_scaler = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        t_scaler = pickle.load(f)
    return f_scaler, t_scaler


def build_model(device, feat_cols):
    targs = resolved_target_cols(cfg)
    mt = cfg.model_type.lower()

    if mt in ("lstm", "gru"):
        model = RNNRegressor(
            input_size=len(feat_cols),
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
            num_inputs=len(feat_cols),
            num_channels=list(cfg.tcn_channels),
            kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            out_dim=len(targs),
        )

    return model.to(device)



def predict_scenario(csv_path, ckpt_path, tag):
    print(f"\n=== Scenario inference for {csv_path} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1) Load raw scenario and build the SAME features as training ---
    df_raw = pd.read_csv(csv_path)
    df_feat = make_features_for_scenario(df_raw, cfg)

    # --- 2) Load scalers from training ---
    f_scaler, t_scaler = load_scalers()

    # How many features was the scaler fitted on?
    n_feat_trained = f_scaler.n_features_in_

    # Full configured feature list (may be longer)
    feat_cols_full = resolved_feature_cols(cfg)

    # Use ONLY the first n_feat_trained features (the ones used during training)
    feat_cols = feat_cols_full[:n_feat_trained]

    print("feat_cols used for inference:", feat_cols)
    print("Scaler expects:", n_feat_trained, "features")
    print("df_feat[feat_cols].shape:", df_feat[feat_cols].shape)

    targ_cols = resolved_target_cols(cfg)

    # --- 3) Apply feature scaling (NO refit!!) ---
    df_scaled = df_feat.copy()
    df_scaled[feat_cols] = f_scaler.transform(df_feat[feat_cols])

    # --- 4) Build windows exactly like training ---
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)

    ds = WindowedDataset(df_scaled, feat_cols, targ_cols, window, stride)
    if len(ds) == 0:
        print("[WARN] No windows for", csv_path)
        return

    # --- 5) Load trained model with matching input dimension ---
    model = build_model(device, feat_cols)
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
    preds[targ_cols] = t_scaler.inverse_transform(
        preds[[t + "_scaled_pred" for t in targ_cols]]
    )

    # pick the PM10 column from targets (e.g. "PM10_ug_per_m3" or residual-shifted)
    pm10_col = [c for c in targ_cols if "PM10" in c][0]

    # --- 8) Align predictions with truth ---
    df_plot = pd.DataFrame({
        "PM10_true": df_feat["PM10_ug_per_m3"].iloc[window - 1:].reset_index(drop=True),
        "PM10_pred": preds[pm10_col].reset_index(drop=True),
    })

    # --- 9) Plot ---
    out_path = f"figures/{Path(csv_path).stem}_{tag}_pm10_overlay.png"
    plot_overlay(
        df_plot,
        out_path,
        title=f"{Path(csv_path).stem} – PM10 true vs predicted ({tag})",
    )

    print("Saved:", out_path)


if __name__ == "__main__":
    booster = "data/processed/booster_clean.csv"
    stopper = "data/processed/stopper_clean.csv"

    ckpt_plain = ".runs/tcn_best.pt"       # or .runs/tcn_plain_best.pt if you renamed
    # ckpt_phys  = ".runs/tcn_phys_best.pt"

    predict_scenario(booster, ckpt_plain, "plain")
    predict_scenario(stopper, ckpt_plain, "plain")

    # if you later have a phys model:
    # predict_scenario(booster, ckpt_phys, "phys")
    # predict_scenario(stopper, ckpt_phys, "phys")
