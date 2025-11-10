# src/report2.py
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Make matplotlib/headless safe on WSL/servers
os.environ.setdefault("MPLBACKEND", "Agg")

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
from src.models import RNNRegressor, TCNRegressor


# Extra plots/utilities you created in report_more.py
from src.report_more import (
    plot_hybrid_vs_physics,        # (aligned_df, pred_df, out_png)
    binned_mae,                    # (y_true, y_pred, bins, out_png)
    brake_onset_average,           # (aligned_df, pred_df, pre_s, post_s, hz) -> DataFrame
    plot_brake_onset_average,      # (avg_df, out_png)
    residual_autocorr,             # (residuals, out_png)
    rolling_mae,                   # (y_true, y_pred, window, out_png)
    permutation_feature_importance # (model, loader, feat_names, out_png)
)

def _prep_splits():
    """
    Use the same pipeline as training: prepare_frames() builds residual/physics
    targets and returns {'train','val','test'} splits already transformed.
    """
    df = pd.read_csv(cfg.data_path)
    return prepare_frames(df)

@torch.no_grad()
def _predict_df(model, ds):
    """
    Return DataFrame with columns: PM10_true/pred, PM2_5_true/pred, PM1_true/pred
    (predictions are in residual space, matching training targets).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    truths, preds = [], []
    device = next(model.parameters()).device

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yp = model(xb)
        truths.append(yb.cpu().numpy())
        preds.append(yp.cpu().numpy())

    if not truths:
        return pd.DataFrame()

    ytrue = np.vstack(truths)
    ypred = np.vstack(preds)

    # Map to base names for plots
    name_map = {"PM10_ug_per_m3": "PM10", "PM2_5_ug_per_m3": "PM2_5", "PM1_ug_per_m3": "PM1"}
    bases = [name_map[c] for c in cfg.target_cols]

    out = {}
    for i, b in enumerate(bases):
        out[f"{b}_true"] = ytrue[:, i]
        out[f"{b}_pred"] = ypred[:, i]
    return pd.DataFrame(out)

def main():
    Path("figures").mkdir(parents=True, exist_ok=True)

    # 1) Prepare data (identical logic to training/evaluate)
    splits = _prep_splits()
    test_df = splits["test"].reset_index(drop=True)

    # Windowing must match training
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    feat_cols = resolved_feature_cols(cfg)          # features used in training
    targ_cols = resolved_target_cols(cfg)           # residual (or physics residual) targets used in training

    test_ds = WindowedDataset(test_df, feat_cols, targ_cols, window, stride)

        # 2) Load model (match training config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt = cfg.model_type.lower()

    if mt in ("lstm", "gru"):
        model = RNNRegressor(
            input_size=len(feat_cols),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.model_type,          # "lstm" or "gru"
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=len(targ_cols),
        ).to(device)
    elif mt == "tcn":
        model = TCNRegressor(
            input_size=len(feat_cols),
            out_dim=len(targ_cols),
            channels_list=getattr(cfg, "tcn_channels", (64, 64, 64)),
            kernel_size=getattr(cfg, "tcn_kernel_size", 3),
            dropout=cfg.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown cfg.model_type={cfg.model_type}")

    ckpt = f".runs/{cfg.model_type}_best.pt"   # e.g. .runs/tcn_best.pt
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 3) Predictions in residual space
    pred_df = _predict_df(model, test_ds)
    if pred_df.empty:
        print("[report2] No test windows; nothing to plot.")
        return

    # Align regime/aux columns with end-of-window targets
    aligned = test_df.iloc[window : window + len(pred_df)].reset_index(drop=True)

    # --- Extra figures ---

    # 3a) Binned MAE vs speed for PM2.5 residuals (use positional args)
    binned_mae(
    pred_df["PM2_5_true"].values,
    pred_df["PM2_5_pred"].values,
    aligned["speed_kmh"].values,
    20,  # number of bins
    "figures/fig_binned_mae_pm25_speed.png",
    )


    # 3b) Residual autocorrelation (PM2.5 residuals)
    residual_autocorr(
        pred_df["PM2_5_true"].values - pred_df["PM2_5_pred"].values,
        "figures/fig_acf_pm25_resid.png",
    )

    # 3c) Rolling MAE (example: 10-minute window @ 5 Hz = 3000 samples)
    rolling_mae(
        pred_df["PM2_5_true"].values,
        pred_df["PM2_5_pred"].values,
        3000,
        "figures/fig_rolling_mae_pm25.png",
    )

    # 3d) Brake-onset average residuals
    avg_df = brake_onset_average(aligned, pred_df, 5, 15, cfg.target_hz)  # pre=5s, post=15s
    plot_brake_onset_average(avg_df, "figures/fig_brake_onset_avg_pm25.png")

    # 3e) Hybrid vs physics overlay for PM10 (if physics cols exist)
    try:
        plot_hybrid_vs_physics(aligned, pred_df, "figures/fig_hybrid_vs_physics_pm10.png")
    except Exception as e:
        print("[report2] Skipped hybrid_vs_physics (missing physics columns):", e)

    # 3f) Permutation feature importance (small loader; may be slow)
    try:
        from torch.utils.data import DataLoader
        loader_small = DataLoader(test_ds, batch_size=min(64, cfg.batch_size), shuffle=False)
        permutation_feature_importance(
            model,
            loader_small,
            feat_cols,
            "figures/fig_permutation_importance.png",
        )
    except Exception as e:
        print("[report2] Skipped permutation importance:", e)

    print("[report2] wrote extra figs in 'figures/'")

if __name__ == "__main__":
    main()
