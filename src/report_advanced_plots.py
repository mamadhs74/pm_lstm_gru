# src/report_advanced_plots.py
"""
Advanced, publication-grade diagnostics & plots (no SHAP).

What you get:
  ✓ Error heatmaps (|error| vs speed, brake)
  ✓ Conditional response (data-driven PDP-style) for key drivers
  ✓ Residual temporal autocorrelation (already in report_more)
  ✓ Permutation feature importance (already in report_more)
  ✓ Physics vs Hybrid scatter (if physics baseline enabled)
  ✓ Power-law scaling check (PM vs power proxy) — optional
  ✓ Cumulative emissions (time-based) & scenario bars
All figures saved into ./figures/.

Assumptions:
- Uses SAME preprocessing as training (prepare_frames), i.e., features in splits are scaled.
- For plots in physical units, we re-load the raw CSV and align by window end indices.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import (
    prepare_frames, downsample, add_regime_features
)
from src.datasets import WindowedDataset
from src.models import RNNRegressor, TCNRegressor
from src.visualize import plot_overlay
from src.report_more import (
    residual_autocorr,
    permutation_feature_importance,
)

# ----------------------------
# Helpers
# ----------------------------

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _build_model(input_size: int, out_dim: int, device):
    mt = cfg.model_type.lower()
    if mt in ("lstm", "gru"):
        m = RNNRegressor(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=mt,
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=out_dim,
        ).to(device)
    elif mt == "tcn":
        m = TCNRegressor(
            input_size=input_size,
            out_dim=out_dim,
            channels_list=getattr(cfg, "tcn_channels", (64, 64, 64)),
            kernel_size=getattr(cfg, "tcn_kernel_size", 3),
            dropout=cfg.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown cfg.model_type={cfg.model_type}")
    return m

def _name_base(col: str) -> str:
    # "PM2_5_ug_per_m3" -> "PM2_5"
    return col.split("_ug")[0]

@torch.no_grad()
def _predict_df(model, ds, batch_size=256):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    ys_t, ys_p = [], []
    dev = next(model.parameters()).device
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        yp = model(xb)
        ys_t.append(yb.cpu().numpy())
        ys_p.append(yp.cpu().numpy())
    if not ys_t:
        return pd.DataFrame()
    ytrue = np.vstack(ys_t)
    ypred = np.vstack(ys_p)
    out = {}
    # We want human-friendly names in cfg.target_cols order
    for i, tcol in enumerate(cfg.target_cols):
        base = _name_base(tcol)
        out[f"{base}_true"] = ytrue[:, i]
        out[f"{base}_pred"] = ypred[:, i]
    return pd.DataFrame(out)

def _safe_hist2d(x, y, bins=24, vmin=None, vmax=None, cmap="viridis", xlabel="", ylabel="", title="", out_png="figures/tmp.png"):
    plt.figure(figsize=(6,4))
    h = plt.hist2d(x, y, bins=bins, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    cbar = plt.colorbar(h[3]); cbar.set_label("count")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _binned_stat(x, y, bins, func=np.mean):
    # compute func(y) per bin of x
    x = np.asarray(x, float); y = np.asarray(y, float)
    edges = np.linspace(np.nanmin(x), np.nanmax(x)+1e-12, bins+1) if np.nanmin(x)!=np.nanmax(x) else np.linspace(0,1,bins+1)
    idx = np.digitize(x, edges) - 1
    centers = 0.5*(edges[:-1]+edges[1:])
    vals = np.full(bins, np.nan)
    errs = np.full(bins, np.nan)
    for b in range(bins):
        mask = (idx==b) & np.isfinite(y)
        if mask.any():
            yy = y[mask]
            vals[b] = func(yy)
            # stderr as simple variability indicator
            errs[b] = np.std(yy)/np.sqrt(len(yy))
    return centers, vals, errs

def _line_with_error(x, y, e, xlabel, ylabel, title, out_png):
    plt.figure(figsize=(6,4))
    plt.plot(x, y, lw=2)
    if np.any(np.isfinite(e)):
        plt.fill_between(x, y-e, y+e, alpha=0.25)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

# ----------------------------
# Main
# ----------------------------

def main():
    Path("figures").mkdir(parents=True, exist_ok=True)

    # 1) LOAD DATA — two views
    # a) "Model view": scaled features, residual targets (exactly as training)
    raw_model = pd.read_csv(cfg.data_path).dropna(subset=cfg.target_cols)
    splits = prepare_frames(raw_model)
    test_df = splits["test"].reset_index(drop=True)

    # b) "Physical view": re-load raw CSV and downsample, add regimes, but DO NOT scale
    raw_phys = pd.read_csv(cfg.data_path)
    phys_df = add_regime_features(downsample(raw_phys, cfg.orig_hz, cfg.target_hz)).reset_index(drop=True)

    # 2) DATASET & MODEL
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    feats = resolved_feature_cols(cfg)
    targs = resolved_target_cols(cfg)
    test_ds = WindowedDataset(test_df, feats, targs, window, stride)

    device = _device()
    model = _build_model(input_size=len(feats), out_dim=len(targs), device=device)
    ckpt = f".runs/{cfg.model_type}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 3) PREDICT (residual space); align indices to physical view
    pred_df = _predict_df(model, test_ds, batch_size=cfg.batch_size)
    if pred_df.empty:
        print("[report_advanced] no test windows; aborting.")
        return
    end_idx = [e-1 for (s,e) in test_ds.indices]  # window-end alignment
    aligned_model = test_df.iloc[end_idx].reset_index(drop=True)   # scaled features!
    aligned_phys  = phys_df.iloc[end_idx].reset_index(drop=True)   # physical units!

    # If physics baseline columns exist (cfg.use_physics_baseline=True), copy them in
    for tcol in cfg.target_cols:
        base = _name_base(tcol)
        phys_col = f"{base}_phys"
        if phys_col in aligned_model.columns:
            pred_df[phys_col] = aligned_model[phys_col].to_numpy(float)
            pred_df[f"{base}_pred_total"] = pred_df[f"{base}_pred"].to_numpy(float) + pred_df[phys_col].to_numpy(float)

    # ---------------------------------------------------------
    # A) ERROR HEATMAPS — |error| vs speed_kmh & brake_bar (physical)
    # ---------------------------------------------------------
    for base in [_name_base(c) for c in cfg.target_cols]:
        err = np.abs(pred_df[f"{base}_true"].values - pred_df[f"{base}_pred"].values)
        # Use physical columns if present
        spd = aligned_phys["speed_kmh"].values if "speed_kmh" in aligned_phys.columns else aligned_model.get("speed_kmh", pd.Series(0)).values
        hb  = aligned_phys["brake_bar"].values if "brake_bar" in aligned_phys.columns else aligned_model.get("brake_bar", pd.Series(0)).values
        # Heatmap: x=speed, y=brake, color=|error|
        # We'll plot 2D density of (speed, brake) for the worst 20% errors to show hotspots
        q80 = np.nanpercentile(err, 80)
        mask = err >= q80
        _safe_hist2d(
            x=spd[mask], y=hb[mask], bins=28,
            xlabel="Speed [km/h]", ylabel="Brake pressure [bar]",
            title=f"High-error density: {base} (worst 20%)",
            out_png=f"figures/{base.lower()}_error_heatmap.png"
        )

    # ---------------------------------------------------------
    # B) PDP-LIKE (DATA-DRIVEN) CURVES — conditional averages
    #     (no SHAP; we compute E[PM|bin(by)] from test set)
    # ---------------------------------------------------------
    for base in [_name_base(c) for c in cfg.target_cols]:
        # choose hybrid pred if available, else residual pred
        ypred = pred_df.get(f"{base}_pred_total", pred_df[f"{base}_pred"]).values
        ytrue = pred_df[f"{base}_true"].values
        # 1) vs brake_bar
        if "brake_bar" in aligned_phys.columns:
            x = aligned_phys["brake_bar"].values
            xb, yb, eb = _binned_stat(x, ypred, bins=24, func=np.mean)
            _, yt, et = _binned_stat(x, ytrue, bins=24, func=np.mean)
            # plot prediction
            _line_with_error(
                xb, yb, eb,
                xlabel="Brake pressure [bar]",
                ylabel=f"{base} (residual/hybrid units)",
                title=f"Conditional response: {base} vs brake_bar (prediction)",
                out_png=f"figures/{base.lower()}_condresp_brake_pred.png"
            )
            # plot truth
            _line_with_error(
                xb, yt, et,
                xlabel="Brake pressure [bar]",
                ylabel=f"{base} (residual units)",
                title=f"Conditional response: {base} vs brake_bar (truth)",
                out_png=f"figures/{base.lower()}_condresp_brake_true.png"
            )
        # 2) vs speed
        if "speed_kmh" in aligned_phys.columns:
            x = aligned_phys["speed_kmh"].values
            xb, yb, eb = _binned_stat(x, ypred, bins=24, func=np.mean)
            _, yt, et = _binned_stat(x, ytrue, bins=24, func=np.mean)
            _line_with_error(
                xb, yb, eb,
                xlabel="Speed [km/h]",
                ylabel=f"{base} (residual/hybrid units)",
                title=f"Conditional response: {base} vs speed (prediction)",
                out_png=f"figures/{base.lower()}_condresp_speed_pred.png"
            )
            _line_with_error(
                xb, yt, et,
                xlabel="Speed [km/h]",
                ylabel=f"{base} (residual units)",
                title=f"Conditional response: {base} vs speed (truth)",
                out_png=f"figures/{base.lower()}_condresp_speed_true.png"
            )

    # ---------------------------------------------------------
    # C) RESIDUAL TEMPORAL ACF — checks memory mismatch
    # ---------------------------------------------------------
    for base in [_name_base(c) for c in cfg.target_cols]:
        res = pred_df[f"{base}_true"].values - pred_df[f"{base}_pred"].values
        residual_autocorr(res, f"figures/{base.lower()}_residual_acf.png")

    # ---------------------------------------------------------
    # D) PERMUTATION FEATURE IMPORTANCE — already implemented
    #     (works in scaled feature space, which matches the model)
    # ---------------------------------------------------------
    try:
        from torch.utils.data import DataLoader
        loader_small = DataLoader(test_ds, batch_size=min(64, cfg.batch_size), shuffle=False)
        permutation_feature_importance(
            model=model,
            loader=loader_small,
            feat_names=feats,
            out_png="figures/permutation_importance.png"
        )
    except Exception as e:
        print("[report_advanced] Skipped permutation importance:", e)

    # ---------------------------------------------------------
    # E) Physics vs Hybrid scatter (if physics columns present)
    # ---------------------------------------------------------
    any_phys = any(f"{_name_base(c)}_phys" in pred_df.columns for c in cfg.target_cols)
    if any_phys:
        for tcol in cfg.target_cols:
            base = _name_base(tcol)
            phys_col = f"{base}_phys"
            if phys_col not in pred_df.columns:
                continue
            # residual truth → physical truth
            y_true_tot = pred_df[phys_col].to_numpy(float) + pred_df[f"{base}_true"].to_numpy(float)
            y_pred_tot = pred_df[phys_col].to_numpy(float) + pred_df[f"{base}_pred"].to_numpy(float)
            plt.figure(figsize=(4.8,4.8))
            plt.scatter(y_true_tot, y_pred_tot, s=6, alpha=0.4)
            lims = [np.nanmin(y_true_tot), np.nanmax(y_true_tot)]
            if np.isfinite(lims).all():
                plt.plot(lims, lims, 'k--', lw=1)
            plt.xlabel(f"{base} true (physical)"); plt.ylabel(f"{base} pred (hybrid physical)")
            plt.title(f"Hybrid vs truth — {base}")
            plt.tight_layout(); plt.savefig(f"figures/{base.lower()}_hybrid_scatter.png", dpi=160); plt.close()

    # ---------------------------------------------------------
    # F) Power-law scaling check (proxy): PM vs (|a_x| * v)
    #     Uses physical speed and (if available) physical long. accel column;
    #     else derive a_x via finite diff on speed.
    # ---------------------------------------------------------
    if "speed_kmh" in aligned_phys.columns:
        v_mps = aligned_phys["speed_kmh"].to_numpy(float) * (1000.0/3600.0)
        if "acc_long_ms2" in aligned_phys.columns:
            ax = aligned_phys["acc_long_ms2"].to_numpy(float)
        else:
            # rough derivative in physical view
            dt = 1.0 / cfg.target_hz
            ax = np.gradient(v_mps, dt)
        power_proxy = np.abs(ax) * v_mps  # W/kg up to scale
        for tcol in cfg.target_cols:
            base = _name_base(tcol)
            ypred = pred_df.get(f"{base}_pred_total", pred_df[f"{base}_pred"]).values
            # Bin log-log robustly
            mask = (power_proxy > 1e-4) & np.isfinite(ypred)
            xp = np.log10(power_proxy[mask])
            yp = np.log10(np.abs(ypred[mask])+1e-6)
            if xp.size > 20:
                # fit slope
                A = np.vstack([xp, np.ones_like(xp)]).T
                slope, intercept = np.linalg.lstsq(A, yp, rcond=None)[0]
                plt.figure(figsize=(5,4))
                plt.scatter(xp, yp, s=4, alpha=0.25)
                xr = np.array([np.nanmin(xp), np.nanmax(xp)])
                plt.plot(xr, slope*xr + intercept, 'r-', lw=2, label=f"α ≈ {slope:.2f}")
                plt.xlabel("log10(|a_x|·v) [arb.]")
                plt.ylabel(f"log10({base} pred) [arb.]")
                plt.title(f"Scaling check: {base} vs power proxy")
                plt.legend()
                plt.tight_layout(); plt.savefig(f"figures/{base.lower()}_scaling_power.png", dpi=160); plt.close()

    # ---------------------------------------------------------
    # G) Cumulative emissions (time-based) — illustrative
    #     Integrate predicted series to show where emissions accrue.
    # ---------------------------------------------------------
    dt = 1.0 / cfg.target_hz
    for tcol in cfg.target_cols:
        base = _name_base(tcol)
        series = pred_df.get(f"{base}_pred_total", pred_df[f"{base}_pred"]).to_numpy(float)
        cum = np.cumsum(np.maximum(series, 0.0) * dt)  # "excess mass" proxy
        plt.figure(figsize=(6,4))
        plt.plot(cum, lw=2)
        plt.xlabel("Window index (aligned with end-of-window)"); plt.ylabel(f"Cumulative {base} (arb. units)")
        plt.title(f"Cumulative predicted {base} (time-integral, ≥0 only)")
        plt.tight_layout(); plt.savefig(f"figures/{base.lower()}_cumulative.png", dpi=160); plt.close()

    print("[report_advanced] Wrote plots into 'figures/'")


if __name__ == "__main__":
    main()
