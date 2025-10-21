# src/evaluate.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import cfg
from src.preprocess import (
    downsample, add_regime_features, leak_free_split,
    fit_transformers, apply_transformers, build_residual_targets, shift_targets
)
from src.datasets import WindowedDataset
from src.models import RNNRegressor
from src.visualize import plot_overlay, plot_parity, plot_residual_hist
from src.diagnostics import regime_metrics


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
def _predict_df(model, ds, device):
    """Return DataFrame with columns: PM10_true/pred, PM2_5_true/pred, PM1_true/pred (residual space)."""
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    truths, preds = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yp = model(xb)
        truths.append(yb.cpu().numpy())
        preds.append(yp.cpu().numpy())

    if not truths:
        return pd.DataFrame()

    ytrue = np.vstack(truths)
    ypred = np.vstack(preds)

    # Map cfg.target_cols -> base names expected by viz/diagnostics
    # Assumes exactly these three targets in order.
    base_names = {
        "PM10_ug_per_m3": "PM10",
        "PM2_5_ug_per_m3": "PM2_5",
        "PM1_ug_per_m3": "PM1",
    }
    bases = [base_names[c] for c in cfg.target_cols]

    out = {}
    for i, base in enumerate(bases):
        out[f"{base}_true"] = ytrue[:, i]
        out[f"{base}_pred"] = ypred[:, i]
    return pd.DataFrame(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = _prep_splits()

    # Build test dataset (windowing identical to train)
    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps
    feat_cols = cfg.feature_cols + ["idle", "brake_event", "dv", "jerk"]
    test_df = splits["test"].reset_index(drop=True)
    test_ds = WindowedDataset(test_df, feat_cols, cfg.target_cols, window, stride)

    # Load best checkpoint
    model = RNNRegressor(
        input_size=len(cfg.feature_cols) + 4,
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

    # Predictions (residual space)
    pred_df = _predict_df(model, test_ds, device)
    if pred_df.empty:
        print("No test windows; nothing to evaluate.")
        return

    # Save predictions
    out_csv = "data/processed/test_predictions_residuals.csv"
    pred_df.to_csv(out_csv, index=False)

    # --- Metrics: overall MAE (residual space) ---
    overall_mae = {
        t: float(np.abs(pred_df[f"{t}_true"] - pred_df[f"{t}_pred"]).mean())
        for t in ["PM10", "PM2_5", "PM1"]
    }
    print("Test MAE (residual):", overall_mae)

        # --- Regime-sliced metrics ---
    # Align regime flags with predictions: offset by the window length (in steps)
    aligned_regimes = test_df.iloc[window : window + len(pred_df)].reset_index(drop=True)

    # Ensure expected regime columns exist for diagnostics
    # Alias brake_event -> brake if needed
    if "brake" not in aligned_regimes.columns and "brake_event" in aligned_regimes.columns:
        aligned_regimes = aligned_regimes.copy()
        aligned_regimes["brake"] = aligned_regimes["brake_event"].astype(int)

    # Ensure 'idle' exists (it should, from add_regime_features)
    if "idle" not in aligned_regimes.columns:
        aligned_regimes = aligned_regimes.copy()
        aligned_regimes["idle"] = 0

    # Create 'cruise' = not idle and not brake
    if "cruise" not in aligned_regimes.columns:
        idle_col = aligned_regimes["idle"].astype(int) if "idle" in aligned_regimes else 0
        brake_col = aligned_regimes["brake"].astype(int) if "brake" in aligned_regimes else 0
        aligned_regimes = aligned_regimes.copy()
        aligned_regimes["cruise"] = ((idle_col == 0) & (brake_col == 0)).astype(int)

    # Merge regimes with predictions for diagnostics
    merged = pd.concat([aligned_regimes, pred_df.reset_index(drop=True)], axis=1)

    print("Regime metrics:", regime_metrics(merged))

    # --- Plots (residual space) ---
    plot_overlay(pred_df, "figures/overlay_test_residuals.png", title="Test — residuals (true vs pred)")
    plot_parity(
        pred_df["PM2_5_true"].values, pred_df["PM2_5_pred"].values,
        "figures/parity_pm25_test.png", title="Parity PM2.5 (residual)"
    )
    plot_residual_hist(
        pred_df["PM2_5_true"].values - pred_df["PM2_5_pred"].values,
        "figures/resid_hist_pm25_test.png", title="Residual Histogram PM2.5 (residual)"
    )

    print("Wrote:")
    print(f" - {out_csv}")
    print(" - figures/overlay_test_residuals.png")
    print(" - figures/parity_pm25_test.png")
    print(" - figures/resid_hist_pm25_test.png")


if __name__ == "__main__":
    main()
