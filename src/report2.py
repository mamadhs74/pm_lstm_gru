# src/report2.py
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Ensure headless plotting
os.environ.setdefault("MPLBACKEND", "Agg")

from src.config import cfg
from src.preprocess import (
    downsample, add_regime_features, leak_free_split,
    fit_transformers, apply_transformers, build_residual_targets, shift_targets
)
from src.datasets import WindowedDataset
from src.models import RNNRegressor

# extra plots/utilities you said you created
from src.report_more import (
    plot_hybrid_vs_physics,        # (df_test, pred_df) -> fig
    binned_mae,                    # (y_true, y_pred, bins, out_png)
    brake_onset_average,           # (df_test, pred_df) -> pd.DataFrame
    plot_brake_onset_average,      # (avg_df, out_png)
    residual_autocorr,             # (residuals, out_png)
    rolling_mae,                   # (y_true, y_pred, window, out_png)
    permutation_feature_importance # (model, loader, feat_names, out_png)
)

def _prep_splits():
    df = pd.read_csv(cfg.data_path)
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    splits = leak_free_split(df, cfg)
    scaler = fit_transformers(splits["train"], cfg)
    for k in splits:
        splits[k] = apply_transformers(splits[k], scaler, cfg)
        splits[k] = build_residual_targets(splits[k], cfg)
        splits[k] = shift_targets(splits[k], cfg)
    return splits

@torch.no_grad()
def _predict_df(model, ds):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    truths, preds = [], []
    for xb, yb in loader:
        xb = xb.to(next(model.parameters()).device)
        yb = yb.to(xb.device)
        yp = model(xb)
        truths.append(yb.cpu().numpy())
        preds.append(yp.cpu().numpy())
    if not truths:
        return pd.DataFrame()
    ytrue = np.vstack(truths)
    ypred = np.vstack(preds)
    # map to nice column names (assumes cfg.target_cols order)
    name_map = {"PM10_ug_per_m3":"PM10", "PM2_5_ug_per_m3":"PM2_5", "PM1_ug_per_m3":"PM1"}
    bases = [name_map[c] for c in cfg.target_cols]
    out = {}
    for i, b in enumerate(bases):
        out[f"{b}_true"] = ytrue[:, i]
        out[f"{b}_pred"] = ypred[:, i]
    return pd.DataFrame(out)

def main():
    Path("figures").mkdir(parents=True, exist_ok=True)

    # 1) Prepare data & model (same pipeline as train/evaluate)
    splits = _prep_splits()
    test_df = splits["test"].reset_index(drop=True)

    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps
    feat_cols = cfg.feature_cols + ["idle","brake_event","dv","jerk"]
    test_ds = WindowedDataset(test_df, feat_cols, [f"{c}_resid_shift" for c in cfg.target_cols], window, stride)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNRegressor(
        input_size=len(feat_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,
        conv_frontend=cfg.conv_frontend,
        conv_channels=cfg.conv_channels,
        out_dim=len(cfg.target_cols),
    ).to(device)
    ckpt = f".runs/{cfg.model_type}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 2) Predictions in residual space (like evaluate)
    pred_df = _predict_df(model, test_ds)
    if pred_df.empty:
        print("[report2] No test windows; nothing to plot.")
        return

    # Align regime flags with the end-of-window targets
    aligned = test_df.iloc[window : window + len(pred_df)].reset_index(drop=True)

    # 3) Extra figures
    # 3a) Binned MAE vs speed for PM2.5 residuals
    binned_mae(
        y_true = pred_df["PM2_5_true"].values,
        y_pred = pred_df["PM2_5_pred"].values,
        bins   = aligned["speed_kmh"].values,
        out_png= "figures/fig_binned_mae_pm25_speed.png"
    )

    # 3b) Residual autocorrelation for PM2.5
    residual_autocorr(
        residuals = pred_df["PM2_5_true"].values - pred_df["PM2_5_pred"].values,
        out_png   = "figures/fig_acf_pm25_resid.png"
    )

    # 3c) Rolling MAE (10-minute window @ 5 Hz ~ 3000 samples; adjust)
    rolling_mae(
        y_true = pred_df["PM2_5_true"].values,
        y_pred = pred_df["PM2_5_pred"].values,
        window = 3000,
        out_png= "figures/fig_rolling_mae_pm25.png"
    )

    # 3d) Brake-onset average residuals
    avg_df = brake_onset_average(aligned, pred_df, pre_s=5, post_s=15, hz=cfg.target_hz)
    plot_brake_onset_average(avg_df, "figures/fig_brake_onset_avg_pm25.png")

    # 3e) Hybrid vs physics overlay for PM10 (if you also saved physics baseline)
    try:
        plot_hybrid_vs_physics(aligned, pred_df, out_png="figures/fig_hybrid_vs_physics_pm10.png")
    except Exception as e:
        print("[report2] Skipped hybrid_vs_physics (missing physics columns):", e)

    # 3f) Permutation feature importance (optional; may be slow)
    try:
        from torch.utils.data import DataLoader
        loader_small = DataLoader(test_ds, batch_size=min(64, cfg.batch_size), shuffle=False)
        permutation_feature_importance(
            model   = model,
            loader  = loader_small,
            feat_names = feat_cols,
            out_png = "figures/fig_permutation_importance.png"
        )
    except Exception as e:
        print("[report2] Skipped permutation importance:", e)

    print("[report2] wrote extra figs in 'figures/'")

if __name__ == "__main__":
    main()
