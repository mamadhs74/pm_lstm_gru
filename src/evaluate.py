# src/evaluate.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
from src.models import RNNRegressor
from src.visualize import plot_overlay, plot_parity, plot_residual_hist
from src.diagnostics import regime_metrics

def _base_name(col: str) -> str:
    # "PM10_ug_per_m3" -> "PM10"
    return col.split("_ug")[0]

@torch.no_grad()
def _predict_df(model, ds, device, display_names, test_df_for_phys=None):
    """
    Returns a DataFrame with columns (residual space):
      PM10_true, PM10_pred, PM2_5_true, PM2_5_pred, PM1_true, PM1_pred

    If physics columns exist in test_df_for_phys (e.g., 'PM10_phys'),
    also returns PM10_phys and PM10_pred_total = PM10_phys + PM10_pred (hybrid).
    """
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    truths, preds = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yp = model(xb)
        truths.append(yb.cpu().numpy())
        preds.append(yp.cpu().numpy())

    if not truths:
        return pd.DataFrame()

    ytrue = np.vstack(truths)  # residual ground truth (shifted target)
    ypred = np.vstack(preds)   # residual prediction

    out = {}
    for i, disp in enumerate(display_names):
        base = _base_name(disp)  # PM10 / PM2_5 / PM1
        out[f"{base}_true"] = ytrue[:, i]
        out[f"{base}_pred"] = ypred[:, i]

    df_pred = pd.DataFrame(out)

    # If we have physics baselines for the exact rows (window end indices), add hybrid columns
    if test_df_for_phys is not None:
        # WindowedDataset targets are taken at the window end index (e-1)
        end_idxs = [e - 1 for (s, e) in ds.indices]
        phys_cols = [f"{_base_name(t)}_phys" for t in display_names]
        have_all_phys = all(c in test_df_for_phys.columns for c in phys_cols)
        if have_all_phys:
            for t in display_names:
                base = _base_name(t)
                phys = test_df_for_phys.loc[end_idxs, f"{base}_phys"].to_numpy(dtype=float)
                df_pred[f"{base}_phys"] = phys
                df_pred[f"{base}_pred_total"] = phys + df_pred[f"{base}_pred"].to_numpy(dtype=float)
    return df_pred

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Use the SAME preprocessing as training
    raw = pd.read_csv(cfg.data_path).dropna(subset=cfg.target_cols)
    splits = prepare_frames(raw)

    # Build test dataset (windowing identical to train)
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)

    feats = resolved_feature_cols(cfg)   # feature cols incl. regimes
    targs = resolved_target_cols(cfg)    # residual target column names (EMA or physics)

    test_df = splits["test"].reset_index(drop=True)
    test_ds = WindowedDataset(test_df, feats, targs, window, stride)

    # Load best checkpoint
    model = RNNRegressor(
        input_size=len(feats),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,
        conv_frontend=cfg.conv_frontend,
        conv_channels=cfg.conv_channels,
        out_dim=len(targs),
    ).to(device)
    ckpt = f".runs/{cfg.model_type}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Decide the human-friendly display order (corresponds to cfg.target_cols)
    # even though the model is trained on residual columns (targs)
    display_names = cfg.target_cols  # keep PM10/PM2_5/PM1 naming for plots

    # Predict in residual space; if physics columns present in test_df, also make hybrid columns
    # Physics columns exist iff cfg.use_physics_baseline=True and your prepare_frames added *_phys.
    pred_df = _predict_df(model, test_ds, device, display_names, test_df_for_phys=test_df)
    if pred_df.empty:
        print("No test windows; nothing to evaluate.")
        return

    # Save predictions
    out_csv = "data/processed/test_predictions_residuals.csv"
    pred_df.to_csv(out_csv, index=False)

    # --- Metrics: overall MAE (residual space) ---
    overall_mae = {
        _base_name(t): float(np.abs(pred_df[f"{_base_name(t)}_true"] - pred_df[f"{_base_name(t)}_pred"]).mean())
        for t in display_names
    }
    print("Test MAE (residual):", overall_mae)

    # --- Regime-sliced metrics ---
    # Align regime flags with predictions using window end indices
    end_idxs = [e - 1 for (s, e) in test_ds.indices]
    aligned = test_df.iloc[end_idxs].reset_index(drop=True).copy()

    # Diagnostics expect 'brake', 'idle', and 'cruise'
    if "brake" not in aligned.columns and "brake_event" in aligned.columns:
        aligned["brake"] = aligned["brake_event"].astype(int)
    if "idle" not in aligned.columns:
        aligned["idle"] = 0
    if "cruise" not in aligned.columns:
        aligned["cruise"] = ((aligned["idle"].astype(int) == 0) &
                             (aligned["brake"].astype(int) == 0)).astype(int)

    merged = pd.concat([aligned, pred_df.reset_index(drop=True)], axis=1)
    print("Regime metrics:", regime_metrics(merged))

    # --- Plots (residual space) ---
    # Your overlay expects columns {PM*_true, PM*_pred}
    plot_overlay(pred_df, "figures/overlay_test_residuals.png", title="Test — residuals (true vs pred)")

    # Parity & histogram for PM2.5 residuals
    pm2 = "PM2_5"
    plot_parity(
        pred_df[f"{pm2}_true"].values,
        pred_df[f"{pm2}_pred"].values,
        "figures/parity_pm25_test.png",
        title="Parity PM2.5 (residual)"
    )
    plot_residual_hist(
        pred_df[f"{pm2}_true"].values - pred_df[f"{pm2}_pred"].values,
        "figures/resid_hist_pm25_test.png",
        title="Residual Histogram PM2.5 (residual)"
    )

    print("Wrote:")
    print(f" - {out_csv}")
    print(" - figures/overlay_test_residuals.png")
    print(" - figures/parity_pm25_test.png")
    print(" - figures/resid_hist_pm25_test.png")

    # --- Optional: if hybrid columns exist, you can add more plots here
    # Example: overlay total prediction vs truth in *physical* space for PM10
    base = "PM10"
    if f"{base}_pred_total" in pred_df.columns:
        df_hybrid = pd.DataFrame({
            f"{base}_true": pred_df[f"{base}_true"].values + pred_df[f"{base}_phys"].values,  # convert residual truth -> physical truth
            f"{base}_pred": pred_df[f"{base}_pred_total"].values
        })
        plot_overlay(df_hybrid, "figures/overlay_test_pm10_hybrid.png",
                     title="Test — PM10 total (hybrid vs truth)")
        print(" - figures/overlay_test_pm10_hybrid.png")

if __name__ == "__main__":
    main()
