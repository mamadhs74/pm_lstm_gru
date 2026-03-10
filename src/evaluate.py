# src/evaluate.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
# top of file: add TCNRegressor to the import
from src.models import RNNRegressor, TCNRegressor
from src.visualize import plot_overlay, plot_parity, plot_residual_hist
from src.diagnostics import regime_metrics

def _base_name(col: str) -> str:
    # "PM10_ug_per_m3" -> "PM10", "P_VA" -> "P_VA"
    return col.split("_ug")[0] if "_ug" in col else col

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
        if isinstance(yp, (tuple, list)):
            yp = yp[0]
        truths.append(yb.detach().cpu().numpy())
        preds.append(yp.detach().cpu().numpy())


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
                phys = test_df_for_phys.iloc[end_idxs][f"{base}_phys"].to_numpy(dtype=float)
                df_pred[f"{base}_phys"] = phys
                df_pred[f"{base}_pred_total"] = phys + df_pred[f"{base}_pred"].to_numpy(dtype=float)
    return df_pred

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from pathlib import Path
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # --- Use the SAME preprocessing as training
    raw = pd.read_csv(cfg.data_path).dropna(subset=cfg.target_cols)
    splits = prepare_frames(raw)

    # Build test dataset (windowing identical to train)
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    feat_cols = resolved_feature_cols(cfg)          
    targ_cols = resolved_target_cols(cfg)  
    print("cfg.target_cols:", cfg.target_cols)
    print("resolved_target_cols(cfg):", targ_cols)
    assert len(targ_cols) == len(cfg.target_cols)
    


    test_df = splits["test"].reset_index(drop=True)
    test_ds = WindowedDataset(test_df, feat_cols, targ_cols, window, stride)

        # --- Build the model that matches cfg.model_type and load its checkpoint ---
    feats = resolved_feature_cols(cfg)
    targs = resolved_target_cols(cfg)

    mt = cfg.model_type.lower()
    if mt in ("lstm", "gru"):
        model = RNNRegressor(
            input_size=len(feats),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.model_type,          # "lstm" or "gru"
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=len(targs),
        ).to(device)
    elif mt == "tcn":
        model = TCNRegressor(
            input_size=len(feats),
            out_dim=len(targs),
            channels_list=getattr(cfg, "tcn_channels", (64, 64, 64)),
            kernel_size=getattr(cfg, "tcn_kernel_size", 3),
            dropout=cfg.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown cfg.model_type={cfg.model_type}")

    ckpt = f".runs/{cfg.model_type}_best.pt"   # e.g., .runs/tcn_best.pt
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()


    # Decide the human-friendly display order (corresponds to cfg.target_cols)
    # even though the model is trained on residual columns (targs)
    display_names = cfg.target_cols  # keep PM10/PM2_5/PM1 naming for plots

    # Predict in residual space; if physics columns present in test_df, also make hybrid columns
    # Physics columns exist iff cfg.use_physics_baseline=True and your prepare_frames added *_phys.
    pred_df = _predict_df(model, test_ds, device, display_names, test_df_for_phys=test_df)
        # Inverse-transform predictions back to physical target space when possible
    target_scaler = splits.get("_target_scaler", None)
    pred_phys_df = None

    if target_scaler is not None and len(display_names) == 1:
        base0 = _base_name(display_names[0])

        y_true_scaled = pred_df[f"{base0}_true"].to_numpy().reshape(-1, 1)
        y_pred_scaled = pred_df[f"{base0}_pred"].to_numpy().reshape(-1, 1)

        y_true_phys = target_scaler.inverse_transform(y_true_scaled).ravel()
        y_pred_phys = target_scaler.inverse_transform(y_pred_scaled).ravel()

        pred_phys_df = pd.DataFrame({
            f"{base0}_true": y_true_phys,
            f"{base0}_pred": y_pred_phys,
        })

        mae_phys = float(np.mean(np.abs(y_true_phys - y_pred_phys)))
        rmse_phys = float(np.sqrt(np.mean((y_true_phys - y_pred_phys) ** 2)))

        print(f"Test MAE ({base0}, physical units):", mae_phys)
        print(f"Test RMSE ({base0}, physical units):", rmse_phys)
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
    from src.report_more import residual_autocorr, plot_error_acf_panel


    # pick the first target from config (works for single-target P_VA too)
    t0 = cfg.target_cols[0]
    base0 = _base_name(t0)

    # Residual-space error ACF (always valid)
    residual_autocorr(
        pred_df[f"{base0}_true"].to_numpy() - pred_df[f"{base0}_pred"].to_numpy(),
        f"figures/fig_acf_{base0.lower()}_resid.png",
        max_lag=200,
        title=f"Residual-space error ACF ({base0})"
    )

    # Physics-only vs hybrid error ACF (only if physics columns exist)
    try:
        plot_error_acf_panel(
            aligned_df=aligned,
            pred_df=pred_df,
            out_png=f"figures/fig_acf_{base0.lower()}_error_panel.png",
            target_short=base0,
            max_lag=200
        )
    except Exception as e:
        print(f"[evaluate] Skipped error ACF panel (likely missing physics or truth cols): {e}")
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

    # Scaled-space parity & histogram
    plot_parity(
        pred_df[f"{base0}_true"].values,
        pred_df[f"{base0}_pred"].values,
        f"figures/parity_{base0.lower()}_scaled_test.png",
        title=f"Parity {base0} (scaled)"
    )
    plot_residual_hist(
        pred_df[f"{base0}_true"].values - pred_df[f"{base0}_pred"].values,
        f"figures/resid_hist_{base0.lower()}_scaled_test.png",
        title=f"Residual Histogram {base0} (scaled)"
    )

    # Physical-space parity & histogram if inverse transform was available
    if pred_phys_df is not None:
        plot_parity(
            pred_phys_df[f"{base0}_true"].values,
            pred_phys_df[f"{base0}_pred"].values,
            f"figures/parity_{base0.lower()}_physical_test.png",
            title=f"Parity {base0} (physical units)"
        )
        plot_residual_hist(
            pred_phys_df[f"{base0}_true"].values - pred_phys_df[f"{base0}_pred"].values,
            f"figures/resid_hist_{base0.lower()}_physical_test.png",
            title=f"Residual Histogram {base0} (physical units)"
        )

    print("Wrote:")
    print(f" - {out_csv}")
    print(" - figures/overlay_test_residuals.png")
    print(f" - figures/parity_{base0.lower()}_scaled_test.png")
    print(f" - figures/resid_hist_{base0.lower()}_scaled_test.png")
    if pred_phys_df is not None:
        print(f" - figures/parity_{base0.lower()}_physical_test.png")
        print(f" - figures/resid_hist_{base0.lower()}_physical_test.png")
        
        # Write a diagnostics summary file
    diag_path = f"figures/diagnostics_{base0.lower()}.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Diagnostics ===\n")
        f.write(f"model_type: {cfg.model_type}\n")
        f.write(f"target_cols: {cfg.target_cols}\n")
        f.write(f"resolved_target_cols: {targ_cols}\n")
        f.write(f"resolved_feature_cols: {feat_cols}\n")
        f.write(f"window_s: {cfg.window_s}\n")
        f.write(f"target_hz: {cfg.target_hz}\n")
        f.write(f"stride_steps: {cfg.stride_steps}\n")
        f.write(f"use_physics_baseline: {getattr(cfg, 'use_physics_baseline', None)}\n")
        f.write(f"target_mode: {getattr(cfg, 'target_mode', 'direct')}\n")
        f.write("\n=== Overall MAE (scaled space) ===\n")
        f.write(str(overall_mae) + "\n")
        f.write("\n=== Prediction summary (scaled space) ===\n")
        f.write(pred_df.describe().to_string() + "\n")

        if pred_phys_df is not None:
            f.write("\n=== Prediction summary (physical space) ===\n")
            f.write(pred_phys_df.describe().to_string() + "\n")

        f.write("\n=== Aligned feature summary ===\n")
        cols_present = [c for c in feat_cols if c in aligned.columns]
        if cols_present:
            f.write(aligned[cols_present].describe().T.to_string() + "\n")
            f.write("\n=== Feature unique counts ===\n")
            f.write(aligned[cols_present].nunique().to_string() + "\n")

    print(f" - {diag_path}")
    # --- Optional: Hybrid overlays in PHYSICAL units (physics + residual) ---
    # Works only if physics baseline columns were added by prepare_frames()
    # and copied into pred_df by _predict_df(..., test_df_for_phys=test_df).
    if any(f"{_base_name(t)}_phys" in pred_df.columns for t in display_names):
        for t in display_names:
            base = _base_name(t)  # "PM10", "PM2_5", "PM1"
            phys_col = f"{base}_phys"
            if phys_col in pred_df.columns:
                y_phys = pred_df[phys_col].to_numpy(dtype=float)
                y_true_tot = y_phys + pred_df[f"{base}_true"].to_numpy(dtype=float)  # residual truth -> physical
                y_pred_tot = y_phys + pred_df[f"{base}_pred"].to_numpy(dtype=float)  # residual pred  -> physical

                df_hybrid = pd.DataFrame({f"{base}_true": y_true_tot, f"{base}_pred": y_pred_tot})
                out_png = f"figures/overlay_test_{base.lower()}_hybrid.png"
                plot_overlay(df_hybrid, out_png, title=f"Test — {base} total (hybrid vs truth)")
                print(f" - {out_png}")



if __name__ == "__main__":
    main()
