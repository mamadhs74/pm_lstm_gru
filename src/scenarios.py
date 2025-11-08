# src/scenarios.py
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Headless plots
os.environ.setdefault("MPLBACKEND", "Agg")

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
from src.models import RNNRegressor
from src.visualize import plot_overlay
from src.report_more import binned_mae, residual_autocorr, brake_onset_average, plot_brake_onset_average

def _area_under_curve(y: np.ndarray, dt: float) -> float:
    # np.trapz is deprecated; use trapezoid
    return float(np.trapezoid(y, dx=dt))

@torch.no_grad()
def _predict_df(model, ds, device, display_names, test_df_for_phys=None):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    truths, preds = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yp = model(xb)
        truths.append(yb.cpu().numpy())
        preds.append(yp.cpu().numpy())
    if not truths:
        return pd.DataFrame()
    ytrue = np.vstack(truths); ypred = np.vstack(preds)
    out = {}
    for i, disp in enumerate(display_names):
        base = disp.split("_ug")[0]
        out[f"{base}_true"] = ytrue[:, i]
        out[f"{base}_pred"] = ypred[:, i]
    df_pred = pd.DataFrame(out)
    if test_df_for_phys is not None:
        end_idxs = [e - 1 for (s, e) in ds.indices]
        for t in display_names:
            base = t.split("_ug")[0]
            phys_col = f"{base}_phys"
            if phys_col in test_df_for_phys.columns:
                phys = test_df_for_phys.loc[end_idxs, phys_col].to_numpy(dtype=float)
                df_pred[phys_col] = phys
                df_pred[f"{base}_pred_total"] = phys + df_pred[f"{base}_pred"].to_numpy(dtype=float)
    return df_pred

def _slice_episodes(df, predicate, pre_s=20, post_s=10, hz=5):
    """Return list of (start_idx, end_idx) windows around times where predicate(row) is True."""
    idxs = df.index[predicate(df)].to_numpy()
    k_pre, k_post = int(pre_s*hz), int(post_s*hz)
    episodes = []
    n = len(df)
    for i in idxs:
        s = max(0, i - k_pre); e = min(n, i + k_post)
        if e > s + 5:
            episodes.append((s, e))
    return episodes

def _pick_light_cases(df):
    """
    Heuristic selectors for two light-approach behaviors.
      - accelerate-through: speed rising, no full stop, brake_event low
      - brake-to-stop: clear braking then speed near zero soon after
    """
    v = df["speed_kmh"].astype(float).fillna(0.0).to_numpy()
    b = df.get("brake_event", pd.Series(0, index=df.index)).astype(int).to_numpy()

    dv = np.r_[0.0, np.diff(v)]

    # thresholds (tune as needed)
    accel_like   = (dv > 0.2) & (v > 5.0) & (b == 0)
    braking_like = (b == 1) | (dv < -0.4)
    near_stop    = (v <= 1.0)

    accel_trigger = (pd.Series(accel_like).rolling(5, min_periods=1).mean().to_numpy() > 0.6)
    brake_trigger = (pd.Series(braking_like).rolling(5, min_periods=1).mean().to_numpy() > 0.6)

    ns_next = (pd.Series(near_stop)
                 .rolling(25, min_periods=1).max()
                 .shift(-5)
                 .fillna(False)
                 .astype(bool)
                 .to_numpy())

    brake_to_stop_trigger = brake_trigger & ns_next
    return accel_trigger.astype(bool), brake_to_stop_trigger.astype(bool)

def main():
    Path("figures").mkdir(exist_ok=True, parents=True)

    # 1) Load & preprocess (exactly like training)
    raw = pd.read_csv(cfg.data_path).dropna(subset=cfg.target_cols)
    splits = prepare_frames(raw)
    test_df = splits["test"].reset_index(drop=True)

    # 2) Build dataset & load model
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)
    test_ds = WindowedDataset(test_df, feat_cols, targ_cols, window, stride)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNRegressor(
        input_size=len(feat_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,
        conv_frontend=cfg.conv_frontend,
        conv_channels=cfg.conv_channels,
        out_dim=len(targ_cols),
    ).to(device)
    ckpt = f".runs/{cfg.model_type}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 3) Predict residual (and hybrid if phys columns exist)
    display_names = cfg.target_cols
    pred_df = _predict_df(model, test_ds, device, display_names, test_df_for_phys=test_df)
    if pred_df.empty:
        print("[scenarios] No test windows; nothing to analyze.")
        return

    # Align aux/regime cols with end-of-window indices
    end_idxs = [e - 1 for (s, e) in test_ds.indices]
    aligned = test_df.iloc[end_idxs].reset_index(drop=True)

    # 4) Define two traffic-light behaviors
    accel_trigger, brake_to_stop_trigger = _pick_light_cases(aligned)

    # Build episodes (−20s … +10s around trigger)
    accel_eps = _slice_episodes(aligned, lambda d: accel_trigger, 20, 10, cfg.target_hz)
    brake_eps = _slice_episodes(aligned, lambda d: brake_to_stop_trigger, 20, 10, cfg.target_hz)

    # 5) KPIs (AUC + peak) for PM2.5 series
    base = "PM2_5"
    dt = 1.0 / cfg.target_hz

    def kpi_for_eps(eps, label):
        if not eps:
            print(f"[{label}] N=0  AUC(mean)=n/a  Peak(mean)=n/a")
            return
        aucs, peaks = [], []
        for (s, e) in eps:
            if f"{base}_pred_total" in pred_df.columns and f"{base}_phys" in pred_df.columns:
                series = pred_df.loc[s:e-1, f"{base}_pred_total"].to_numpy(float)
            else:
                series = pred_df.loc[s:e-1, f"{base}_pred"].to_numpy(float)
            aucs.append(_area_under_curve(series, dt))
            peaks.append(float(np.nanmax(series)) if len(series) else np.nan)
        print(f"[{label}] N={len(eps)}  AUC(mean)={np.nanmean(aucs):.3f}  Peak(mean)={np.nanmean(peaks):.3f}")

    kpi_for_eps(accel_eps, "accelerate-through")
    kpi_for_eps(brake_eps, "brake-to-stop")

    # Example overlays
    def overlay_one(ep, out_png):
        s, e = ep
        df_plot = pd.DataFrame({
            f"{base}_true": pred_df.loc[s:e-1, f"{base}_true"].values,
            f"{base}_pred": (pred_df.loc[s:e-1, f"{base}_pred_total"].values
                             if f"{base}_pred_total" in pred_df.columns
                             else pred_df.loc[s:e-1, f"{base}_pred"].values)
        })
        plot_overlay(df_plot, out_png, title=f"{base} — {out_png.split('/')[-1].replace('.png','')}")

    if accel_eps:
        overlay_one(accel_eps[0], "figures/traffic_light_accelerate_pm25.png")
    if brake_eps:
        overlay_one(brake_eps[0], "figures/traffic_light_brake_pm25.png")

    # 6) Sensitivity: PM vs brake pressure (BINNED) — MUST use positional args
    if "brake_bar" in aligned.columns:
        y_pred = (pred_df[f"{base}_pred_total"].values
                  if f"{base}_pred_total" in pred_df.columns else pred_df[f"{base}_pred"].values)
        binned_mae(
            pred_df[f"{base}_true"].values,
            y_pred,
            aligned["brake_bar"].values,
            20,
            "figures/pm25_vs_brake_bar_binned.png",
        )

    # Residual ACF (do we miss dynamics?)
    residual_autocorr(
        pred_df[f"{base}_true"].values - pred_df[f"{base}_pred"].values,
        "figures/pm25_residual_acf.png"
    )

    # Brake-onset average
    avg_df = brake_onset_average(aligned, pred_df, pre_s=5, post_s=15, hz=cfg.target_hz)
    plot_brake_onset_average(avg_df, "figures/pm25_brake_onset_average.png")

    print("[scenarios] wrote figures in 'figures/'")

if __name__ == "__main__":
    main()
