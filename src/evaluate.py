import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
from src.models import RNNRegressor, TCNRegressor
from src.visualize import plot_overlay, plot_parity, plot_residual_hist
from src.diagnostics import regime_metrics


def _base_name(col: str) -> str:
    # "PM10_ug_per_m3" -> "PM10", "P_VA" -> "P_VA"
    return col.split("_ug")[0] if "_ug" in col else col


@torch.no_grad()
def _predict_df(model, ds, device, display_names, test_df_for_phys=None):
    """
    Returns a DataFrame with columns in model-output space:
      <target>_true, <target>_pred

    If physics columns exist in test_df_for_phys, also returns:
      <target>_phys, <target>_pred_total
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

    ytrue = np.vstack(truths)
    ypred = np.vstack(preds)

    out = {}
    for i, disp in enumerate(display_names):
        base = _base_name(disp)
        out[f"{base}_true"] = ytrue[:, i]
        out[f"{base}_pred"] = ypred[:, i]

    df_pred = pd.DataFrame(out)

    if test_df_for_phys is not None:
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
    import matplotlib.pyplot as plt

    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # --- same preprocessing as training ---
    raw = pd.read_csv(cfg.data_path).dropna(subset=cfg.target_cols)
    splits = prepare_frames(raw)

    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    print("cfg.target_cols:", cfg.target_cols)
    print("resolved_target_cols(cfg):", targ_cols)
    assert len(targ_cols) == len(cfg.target_cols)

    test_df = splits["test"].reset_index(drop=True)
    test_ds = WindowedDataset(test_df, feat_cols, targ_cols, window, stride)

    # --- build model exactly like training ---
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

    ckpt = f".runs/{cfg.model_type}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    display_names = cfg.target_cols

    pred_df = _predict_df(model, test_ds, device, display_names, test_df_for_phys=test_df)
    if pred_df.empty:
        print("No test windows; nothing to evaluate.")
        return

    # Inverse-transform model target back to physical units
    # For physics_residual_shift mode, this is the residual first,
    # then we reconstruct total prediction using shifted physics baseline.
    target_scaler = splits.get("_target_scaler", None)
    pred_phys_df = None

    t0 = cfg.target_cols[0]
    base0 = _base_name(t0)

    if target_scaler is not None and len(display_names) == 1:
        y_true_scaled = pred_df[f"{base0}_true"].to_numpy().reshape(-1, 1)
        y_pred_scaled = pred_df[f"{base0}_pred"].to_numpy().reshape(-1, 1)

        y_true_resid = target_scaler.inverse_transform(y_true_scaled).ravel()
        y_pred_resid = target_scaler.inverse_transform(y_pred_scaled).ravel()

        pred_phys_df = pd.DataFrame({
            f"{base0}_true_resid": y_true_resid,
            f"{base0}_pred_resid": y_pred_resid,
        })

        end_idxs = [e - 1 for (s, e) in test_ds.indices]

        phys_shift_col = f"{base0}_phys_shift"
        true_shift_col = f"{t0}_shift"

        if phys_shift_col in test_df.columns and true_shift_col in test_df.columns:
            y_phys = test_df.iloc[end_idxs][phys_shift_col].to_numpy(dtype=float)
            y_true_total = test_df.iloc[end_idxs][true_shift_col].to_numpy(dtype=float)
            y_pred_total = y_phys + y_pred_resid

            pred_phys_df[f"{base0}_phys"] = y_phys
            pred_phys_df[f"{base0}_true"] = y_true_total
            pred_phys_df[f"{base0}_pred"] = y_pred_total

            mae_phys = float(np.mean(np.abs(y_true_total - y_pred_total)))
            rmse_phys = float(np.sqrt(np.mean((y_true_total - y_pred_total) ** 2)))

            print(f"Test MAE ({base0}, physical hybrid units):", mae_phys)
            print(f"Test RMSE ({base0}, physical hybrid units):", rmse_phys)
        else:
            # fallback if physics columns are unavailable
            pred_phys_df[f"{base0}_true"] = y_true_resid
            pred_phys_df[f"{base0}_pred"] = y_pred_resid

            mae_phys = float(np.mean(np.abs(y_true_resid - y_pred_resid)))
            rmse_phys = float(np.sqrt(np.mean((y_true_resid - y_pred_resid) ** 2)))

            print(f"Test MAE ({base0}, fallback physical units):", mae_phys)
            print(f"Test RMSE ({base0}, fallback physical units):", rmse_phys)

    # --- save raw predictions ---
    out_csv = "data/processed/test_predictions_residuals.csv"
    pred_df.to_csv(out_csv, index=False)

    # --- residual/model-space MAE ---
    overall_mae = {
        _base_name(t): float(np.abs(pred_df[f"{_base_name(t)}_true"] - pred_df[f"{_base_name(t)}_pred"]).mean())
        for t in display_names
    }
    print("Test MAE (residual):", overall_mae)

    # --- alignment for diagnostics ---
    end_idxs = [e - 1 for (s, e) in test_ds.indices]
    aligned = test_df.iloc[end_idxs].reset_index(drop=True).copy()

    if f"{base0}_phys" in pred_df.columns:
        phys_abs = np.abs(pred_df[f"{base0}_phys"].to_numpy())
        print(f"[check] {base0}_phys mean abs =", float(np.mean(phys_abs)))

    from src.report_more import (
        residual_autocorr,
        plot_error_acf_panel,
        plot_target_acf,
        plot_target_physics_hybrid_acf,
    )

    residual_autocorr(
        pred_df[f"{base0}_true"].to_numpy() - pred_df[f"{base0}_pred"].to_numpy(),
        f"figures/fig_acf_{base0.lower()}_resid.png",
        max_lag=200,
        title=f"Residual-space error ACF ({base0})"
    )

    try:
        plot_target_acf(
            aligned_df=aligned,
            out_png=f"figures/fig_acf_target_{base0.lower()}.png",
            target_short=base0,
            max_lag=100,
        )
    except Exception as e:
        print(f"[evaluate] Skipped target ACF: {e}")

    try:
        plot_target_physics_hybrid_acf(
            aligned_df=aligned,
            pred_df=pred_df,
            out_png=f"figures/fig_acf_trio_{base0.lower()}.png",
            target_short=base0,
            max_lag=100,
        )
    except Exception as e:
        print(f"[evaluate] Skipped trio ACF plot: {e}")

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

    if "brake" not in aligned.columns and "brake_event" in aligned.columns:
        aligned["brake"] = aligned["brake_event"].astype(int)
    if "idle" not in aligned.columns:
        aligned["idle"] = 0
    if "cruise" not in aligned.columns:
        aligned["cruise"] = (
            (aligned["idle"].astype(int) == 0) &
            (aligned["brake"].astype(int) == 0)
        ).astype(int)

    merged = pd.concat([aligned, pred_df.reset_index(drop=True)], axis=1)
    print("Regime metrics:", regime_metrics(merged))

    # --- existing plots ---
    plot_overlay(pred_df, "figures/overlay_test_residuals.png", title="Test — residuals (true vs pred)")

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

    # --- NEW: report metrics + actual-vs-predicted plot ---
    if pred_phys_df is not None:
        y_true_eval = pred_phys_df[f"{base0}_true"].to_numpy()
        y_pred_eval = pred_phys_df[f"{base0}_pred"].to_numpy()
        metrics_space = "physical"
    else:
        y_true_eval = pred_df[f"{base0}_true"].to_numpy()
        y_pred_eval = pred_df[f"{base0}_pred"].to_numpy()
        metrics_space = "scaled"

    mae_eval = float(mean_absolute_error(y_true_eval, y_pred_eval))
    rmse_eval = float(np.sqrt(mean_squared_error(y_true_eval, y_pred_eval)))
    r2_eval = float(r2_score(y_true_eval, y_pred_eval))

    y_range = float(np.max(y_true_eval) - np.min(y_true_eval))
    nrmse_eval = float(rmse_eval / y_range) if y_range > 0 else np.nan
    bias_eval = float(np.mean(y_pred_eval - y_true_eval))

    metrics_df = pd.DataFrame([{
        "target": base0,
        "space": metrics_space,
        "MAE": mae_eval,
        "RMSE": rmse_eval,
        "NRMSE": nrmse_eval,
        "R2": r2_eval,
        "Bias": bias_eval,
        "N": int(len(y_true_eval)),
    }])
    metrics_csv = f"data/processed/metrics_{base0.lower()}.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_eval, y_pred_eval, alpha=0.5)
    mn = float(min(y_true_eval.min(), y_pred_eval.min()))
    mx = float(max(y_true_eval.max(), y_pred_eval.max()))
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted — {base0} ({metrics_space})")
    plt.tight_layout()
    actual_pred_png = f"figures/actual_vs_predicted_{base0.lower()}.png"
    plt.savefig(actual_pred_png, dpi=150)
    plt.close()

    print(f"MAE ({metrics_space}):", mae_eval)
    print(f"RMSE ({metrics_space}):", rmse_eval)
    print(f"NRMSE ({metrics_space}):", nrmse_eval)
    print(f"R2 ({metrics_space}):", r2_eval)
    print(f"Bias ({metrics_space}):", bias_eval)

    print("Wrote:")
    print(f" - {out_csv}")
    print(" - figures/overlay_test_residuals.png")
    print(f" - figures/parity_{base0.lower()}_scaled_test.png")
    print(f" - figures/resid_hist_{base0.lower()}_scaled_test.png")
    if pred_phys_df is not None:
        print(f" - figures/parity_{base0.lower()}_physical_test.png")
        print(f" - figures/resid_hist_{base0.lower()}_physical_test.png")
    print(f" - {metrics_csv}")
    print(f" - {actual_pred_png}")

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
        f.write("\n=== Overall MAE (scaled/model space) ===\n")
        f.write(str(overall_mae) + "\n")
        f.write("\n=== Report Metrics ===\n")
        f.write(metrics_df.to_string(index=False) + "\n")
        f.write("\n=== Prediction summary (scaled/model space) ===\n")
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

    if any(f"{_base_name(t)}_phys" in pred_df.columns for t in display_names):
        for t in display_names:
            base = _base_name(t)
            phys_col = f"{base}_phys"
            if phys_col in pred_df.columns:
                y_phys = pred_df[phys_col].to_numpy(dtype=float)
                y_true_tot = y_phys + pred_df[f"{base}_true"].to_numpy(dtype=float)
                y_pred_tot = y_phys + pred_df[f"{base}_pred"].to_numpy(dtype=float)

                df_hybrid = pd.DataFrame({
                    f"{base}_true": y_true_tot,
                    f"{base}_pred": y_pred_tot
                })
                out_png = f"figures/overlay_test_{base.lower()}_hybrid.png"
                plot_overlay(df_hybrid, out_png, title=f"Test — {base} total (hybrid vs truth)")
                print(f" - {out_png}")

        # =========================
    # FINAL REPORT METRICS
    # =========================

    if pred_phys_df is not None:
        y_true_eval = pred_phys_df[f"{base0}_true"].to_numpy()
        y_pred_eval = pred_phys_df[f"{base0}_pred"].to_numpy()
        space = "physical"
    else:
        y_true_eval = pred_df[f"{base0}_true"].to_numpy()
        y_pred_eval = pred_df[f"{base0}_pred"].to_numpy()
        space = "scaled"

    # --- metrics ---
    mae = mean_absolute_error(y_true_eval, y_pred_eval)
    rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
    r2 = r2_score(y_true_eval, y_pred_eval)

    # --- NRMSE ---
    y_range = np.max(y_true_eval) - np.min(y_true_eval)
    nrmse = rmse / y_range if y_range > 0 else np.nan

    # --- Bias ---
    bias = np.mean(y_pred_eval - y_true_eval)

    print("\n=== FINAL REPORT METRICS ===")
    print("Space:", space)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("NRMSE:", nrmse)
    print("Bias:", bias)

    # --- SAVE TO CSV ---
    metrics_df = pd.DataFrame([{
        "target": base0,
        "space": space,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NRMSE": nrmse,
        "Bias": bias,
        "N": len(y_true_eval)
    }])

    metrics_path = f"data/processed/metrics_{base0.lower()}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved metrics ->", metrics_path)

if __name__ == "__main__":
    main()