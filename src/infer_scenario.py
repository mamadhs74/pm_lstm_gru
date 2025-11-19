# src/infer_scenario.py

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import (
    canonicalize_units_and_names,
    downsample,
    add_regime_features,
)

from src.visualize import plot_overlay
from src.models import TCNRegressor, RNNRegressor


# -------------------------------------------------------------------
#   Inference Dataset – windows but no targets
# -------------------------------------------------------------------
class InferenceWindowedDataset(Dataset):
    def __init__(self, df, feature_cols, window, stride):
        X = df[feature_cols].values.astype(np.float32)

        self.window = int(window)
        self.stride = int(stride)
        self.X = X

        max_start = len(X) - self.window
        if max_start < 0:
            self.indices = []
        else:
            self.indices = list(range(0, max_start + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return torch.from_numpy(self.X[i : i + self.window])


# -------------------------------------------------------------------
#   Scaler loading
# -------------------------------------------------------------------
def load_scalers():
    with open(".runs/feature_scaler.pkl", "rb") as f:
        f_scaler = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        t_scaler = pickle.load(f)
    return f_scaler, t_scaler


# -------------------------------------------------------------------
#   Build model (compatible with your TCNRegressor)
# -------------------------------------------------------------------
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
        # TCNRegressor __init__(num_inputs, out_dim, channels, kernel_size, dropout)
        model = TCNRegressor(
            len(feat_cols),
            len(targs),
            list(cfg.tcn_channels),
            cfg.tcn_kernel_size,
            cfg.dropout,
        )

    return model.to(device)


# -------------------------------------------------------------------
#   SCENARIO FEATURE PIPELINE
# -------------------------------------------------------------------
def scenario_features(df_raw):
    """Matches exactly the SAME logic as training preprocessing."""
    df = canonicalize_units_and_names(df_raw, cfg)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    return df


# -------------------------------------------------------------------
#   MAIN SCENARIO INFERENCE
# -------------------------------------------------------------------
def predict_scenario(csv_path, ckpt_path, tag):
    print(f"\n=== Scenario inference for {csv_path} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load raw scenario CSV
    df_raw = pd.read_csv(csv_path)

    # 2) Feature engineering (exact training logic)
    df_feat = scenario_features(df_raw)

    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    # Ensure all required features exist
    missing = [c for c in feat_cols if c not in df_feat.columns]
    if missing:
        print("[ERROR] Missing feature columns:", missing)
        return

    # 3) Load scalers from training
    f_scaler, t_scaler = load_scalers()

    # 4) Apply feature scaling
    df_scaled = df_feat.copy()
    df_scaled[feat_cols] = f_scaler.transform(df_feat[feat_cols])

    # 5) Window size selection
    window_raw = int(cfg.window_s * cfg.target_hz)
    window = min(window_raw, len(df_scaled))
    stride = int(cfg.stride_steps)

    print(f"Window requested={window_raw}, scenario={len(df_scaled)}, using window={window}")

    ds = InferenceWindowedDataset(df_scaled, feat_cols, window, stride)
    if len(ds) == 0:
        print("[WARN] No windows – scenario too short.")
        return

    # 6) Load model
    model = build_model(device, feat_cols)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 7) Predict
    preds_scaled = []
    with torch.no_grad():
        for xb in ds:
            xb = xb.unsqueeze(0).to(device)
            out = model(xb)
            preds_scaled.append(out.cpu().numpy().squeeze())

    preds_scaled = np.stack(preds_scaled)

    # 8) Inverse-transform back to real PM space
    preds_real = t_scaler.inverse_transform(preds_scaled)
    # find whichever PM10 target column the model was trained on
    pm10_col = [c for c in targ_cols if c.startswith("PM10_ug_per_m3")][0]
    idx_pm10 = targ_cols.index(pm10_col)

    pm10_pred = preds_real[:, idx_pm10]

    # 9) No true PM available, plot prediction only
    df_plot = pd.DataFrame({
        "PM10_pred": pm10_pred
    })

    print("preds_real[0:20] =", preds_real[:20])

    # 10) Save plot
    out = f"figures/{Path(csv_path).stem}_{tag}_pm10_overlay.png"
    plot_overlay(
        df_plot,
        out,
        title=f"{Path(csv_path).stem} — PM10 predicted ({tag})"
    )

    print("Saved:", out)


# -------------------------------------------------------------------
#   RUN BOTH SCENARIOS
# -------------------------------------------------------------------
if __name__ == "__main__":
    booster = "data/processed/booster_clean.csv"
    stopper = "data/processed/stopper_clean.csv"
    ckpt_plain = ".runs/tcn_best.pt"

    predict_scenario(booster, ckpt_plain, "plain")
    predict_scenario(stopper, ckpt_plain, "plain")
