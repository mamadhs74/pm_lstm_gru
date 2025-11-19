# src/infer_scenario.py
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import canonicalize_units_and_names, engineer_features
from src.visualize import plot_overlay
from src.models import TCNRegressor, RNNRegressor


# ---------- Small dataset for inference ----------

class InferenceWindowedDataset(Dataset):
    """
    Only features, no targets. Each item = one window used for one prediction.
    """
    def __init__(self, df, feature_cols, window, stride):
        self.feature_cols = feature_cols
        X = df[feature_cols].values.astype(np.float32)

        self.window = int(window)
        self.stride = int(stride)
        max_start = len(X) - self.window

        if max_start < 0:
            # not enough samples for even one window
            self.indices = []
            self.X = np.empty((0, self.window, len(feature_cols)), dtype=np.float32)
            return

        self.indices = list(range(0, max_start + 1, self.stride))

        # Pre-store X so __getitem__ is simple
        self.X = X

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i : i + self.window]  # (T, F)
        return torch.from_numpy(x)       # (T, F)


# ---------- Helpers ----------

def load_scalers():
    """Load feature & target scalers saved during training on 1_clean.csv."""
    with open(".runs/feature_scaler.pkl", "rb") as f:
        f_scaler = pickle.load(f)
    with open(".runs/target_scaler.pkl", "rb") as f:
        t_scaler = pickle.load(f)
    return f_scaler, t_scaler


def build_model(device, feat_cols):
    """
    Build the same model architecture as in train.py / scenario_eval.py.
    We use positional args for TCNRegressor to avoid keyword name mismatch.
    """
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
        # TCNRegressor(input_size, out_dim, channels_list, kernel_size, dropout)
        # or TCNRegressor(num_inputs, out_dim, num_channels, kernel_size, dropout)
        # -> positional args keep this robust.
        model = TCNRegressor(
            len(feat_cols),
            len(targs),
            list(cfg.tcn_channels),
            cfg.tcn_kernel_size,
            cfg.dropout,
        )

    return model.to(device)


def make_features_for_scenario(df_raw):
    """
    Apply the same unit canonicalisation + feature engineering as in training,
    but WITHOUT splitting into train/val/test or refitting scalers.
    """
    df = canonicalize_units_and_names(df_raw, cfg)
    df = engineer_features(df, cfg)  # adds idle, dv, jerk, etc., so that
                                     # resolved_feature_cols(cfg) all exist
    return df


def predict_scenario(csv_path: str, ckpt_path: str, tag: str):
    print(f"\n=== Scenario inference for {csv_path} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1) Load raw cleaned scenario ---
    df_raw = pd.read_csv(csv_path)

    # --- 2) Build features exactly as for training ---
    df_feat = make_features_for_scenario(df_raw)

    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    # Safety: check that all feature columns exist
    missing_feats = [c for c in feat_cols if c not in df_feat.columns]
    if missing_feats:
        print("[ERROR] Some required feature columns are missing in scenario:")
        print(missing_feats)
        return

    # --- 3) Load scalers from training on 1_clean.csv ---
    f_scaler, t_scaler = load_scalers()

    # --- 4) Scale features using *training* feature scaler (no refit!) ---
    X_feat = df_feat.copy()
    print("feat_cols used for inference:", feat_cols)
    print("Scaler expects:", f_scaler.n_features_in_, "features")
    print("df_feat[feat_cols].shape:", df_feat[feat_cols].shape)

    X_feat[feat_cols] = f_scaler.transform(df_feat[feat_cols])

    # --- 5) Choose window length: adapt if scenario is shorter than training window ---
    window_raw = int(cfg.window_s * cfg.target_hz)  # e.g. 90 s * 5 Hz = 450
    window = min(window_raw, len(X_feat))
    if window < 5:
        print(f"[WARN] Scenario too short (len={len(X_feat)}) for any window.")
        return

    print(f"window_raw: {window_raw}  -> using window: {window}")

    stride = int(cfg.stride_steps)
    ds = InferenceWindowedDataset(X_feat, feat_cols, window, stride)
    if len(ds) == 0:
        print(f"[WARN] No windows for {csv_path}")
        return

    # --- 6) Build and load trained model (trained on 1_clean.csv via train.py) ---
    model = build_model(device, feat_cols)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- 7) Run inference: one PM vector per window (sequence-to-one) ---
    preds_scaled = []
    with torch.no_grad():
        for xb in ds:
            xb = xb.unsqueeze(0).to(device)   # (1, T, F)
            yb = model(xb)                    # (1, out_dim) last time step
            preds_scaled.append(yb.cpu().numpy().squeeze())

    preds_scaled = np.stack(preds_scaled, axis=0)  # (num_windows, out_dim)

    # --- 8) Inverse-transform predictions back to physical PM (µg/m³ residuals) ---
    preds_phys = t_scaler.inverse_transform(preds_scaled)  # same order as targ_cols
    # Extract PM10 prediction
    idx_pm10 = targ_cols.index("PM10_ug_per_m3")
    pm10_pred = preds_phys[:, idx_pm10]

    # --- 9) Align predictions with timeline (each window -> last index in that window) ---
    idx_end = np.array(ds.indices) + window - 1
    pm10_true = df_feat["PM10_ug_per_m3"].values[idx_end]

    df_plot = pd.DataFrame({
        "PM10_true": pm10_true,
        "PM10_pred": pm10_pred,
    })

    out_path = f"figures/{Path(csv_path).stem}_{tag}_pm10_overlay.png"
    plot_overlay(df_plot, out_path,
                 title=f"{Path(csv_path).stem} – PM10 true vs predicted ({tag})")

    print("Saved:", out_path)


if __name__ == "__main__":
    booster = "data/processed/booster_clean.csv"
    stopper = "data/processed/stopper_clean.csv"

    # model trained with train.py on data/processed/1_clean.csv
    ckpt_plain = ".runs/tcn_best.pt"      # or tcn_plain_best.pt if you renamed it
    # option: a physics model if you trained it
    # ckpt_phys  = ".runs/tcn_phys_best.pt"

    predict_scenario(booster, ckpt_plain, "plain")
    predict_scenario(stopper, ckpt_plain, "plain")

    # If you later train a physics-aware model, you can also call:
    # predict_scenario(booster, ckpt_phys, "phys")
    # predict_scenario(stopper, ckpt_phys, "phys")
