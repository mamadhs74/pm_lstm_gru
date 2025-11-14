# src/report_plots.py
# Professional plots for your EuroBrake-style presentation.
# All patterns mirror official docs:
# - PyTorch eval loop: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# - TensorBoard (already produced by your train.py): https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
# - Matplotlib plotting: https://matplotlib.org/stable/gallery/index.html
# - scikit-learn metrics/patterns (where applicable)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from .preprocess import prepare_frames
from .datasets import WindowedDataset
from src.models import RNNRegressor, TCNRegressor
from .physics import fit_pf_per_target, apply_physics_baseline

# ---------------------------
# Helpers: model + predictions (official PyTorch eval loop pattern)
# ---------------------------

def _build_model(device, out_dim: int, input_size: int):
    """
    Build the SAME model type as in training, with the SAME input_size and out_dim,
    and load the best checkpoint from .runs/.
    """
    if cfg.model_type.lower() in ("lstm", "gru"):
        model = RNNRegressor(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.model_type,
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=out_dim,
        ).to(device)
    elif cfg.model_type.lower() == "tcn":
        model = TCNRegressor(
            input_size=input_size,
            out_dim=out_dim,
            channels_list=getattr(cfg, "tcn_channels", (64, 64, 64)),
            kernel_size=getattr(cfg, "tcn_kernel_size", 3),
            dropout=cfg.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown cfg.model_type={cfg.model_type}")

    ckpt = f".runs/{cfg.model_type}_best.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
@torch.no_grad()
def _predict_loader(model, loader, device):
    """
    Run model on a loader and return (y_true, y_pred) in residual space.

    Supports:
      - deterministic models: model(x) -> y_pred
      - heteroscedastic TCN:  model(x) -> (mu, log_var)
    """
    model.eval()
    ys_true, ys_pred = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)

        # If TCN returns (mu, logv), take mu
        if isinstance(out, (tuple, list)) and len(out) == 2:
            yp = out[0]
        else:
            yp = out

        ys_true.append(yb.cpu().numpy())
        ys_pred.append(yp.cpu().numpy())

    if not ys_true:
        return None, None

    ytrue = np.vstack(ys_true)
    ypred = np.vstack(ys_pred)
    return ytrue, ypred


def _make_val_loader(splits):
    from torch.utils.data import DataLoader

    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps

    # Use exactly the same feature/target sets as in train.py
    feat_cols = resolved_feature_cols(cfg)      # cfg.feature_cols + ["idle","brake_event","dv","jerk"]
    targ_cols = resolved_target_cols(cfg)       # residual target cols

    val_ds = WindowedDataset(
        splits["val"],
        feat_cols,
        targ_cols,
        window,
        stride,
    )

    loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def _align_arrays(*arrs):
    """Align arrays by shortest first dimension (simple, robust)."""
    if not arrs:
        return arrs
    m = min(a.shape[0] for a in arrs)
    return tuple(a[:m] for a in arrs)

# ---------------------------
# FIGURE 1 – Pipeline diagram (matplotlib patches/arrows)
# ---------------------------

def plot_pipeline_diagram(fname="figures/fig1_pipeline.png"):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    def box(x, y, w, h, txt):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                              edgecolor="black", facecolor="white", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center")

    def arrow(x1, y1, x2, y2):
        ar = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, lw=1.2)
        ax.add_patch(ar)

    # Flow: Data -> Physics Baseline -> Residuals -> Windowing/Scaling -> RNN -> Residual PM -> +Baseline -> PM
    box(0.02, 0.35, 0.18, 0.3, "Driving data\n(speed, accel,\n brake, yaw, ...)")
    box(0.235, 0.35, 0.18, 0.3, "Physics\nFeißel PF · W_spec")
    box(0.45, 0.35, 0.18, 0.3, "Residuals\nPM - PF·W_spec")
    box(0.665, 0.35, 0.18, 0.3, "Windowing\n+ Scaling")
    box(0.88, 0.35, 0.18, 0.3, "RNN (GRU/LSTM)\nResidual PM̂")

    arrow(0.20, 0.50, 0.235, 0.50)
    arrow(0.415, 0.50, 0.45, 0.50)
    arrow(0.63, 0.50, 0.665, 0.50)
    arrow(0.845, 0.50, 0.88, 0.50)

    # Add “+ physics” at the end
    ax.text(0.88 + 0.18 + 0.02, 0.50, "+ PF·W_spec → PM̂", va="center")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True, parents=True)
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# FIGURE 4 – Validation overlay (true vs predicted residuals)
# (You already have plot_overlay; we’ll just call it.)
# ---------------------------

def plot_val_overlay_from_loader(model, loader, target_cols, fname="figures/fig4_overlay_val_residuals.png"):
    from .visualize import plot_overlay
    ytrue, ypred = _predict_loader(model, loader, device=torch.device("cpu"))
    if ytrue.shape[0] == 0:
        print("[plot] no validation windows; skipping overlay.")
        return
    # Build df: {PM10_true, PM10_pred, ...} with residuals
    out = pd.DataFrame()
    for i, c in enumerate(target_cols):
        base = c.split("_ug")[0]
        out[f"{base}_true"] = ytrue[:, i]
        out[f"{base}_pred"] = ypred[:, i]
    plot_overlay(out, fname, title="Validation — residuals (true vs pred)")
    print(f"[plot] wrote {fname}")

# ---------------------------
# Physics helpers for Fig 5 & 6
# ---------------------------

def _augment_with_physics(df, pf=None):
    # apply_physics_baseline adds W_spec and *_resid_phys using cfg target names
    if pf is None:
        # Should be fitted on train
        raise ValueError("PF must be provided (fit_pf_per_target on train).")
    return apply_physics_baseline(df, cfg, pf)

def _val_alignment_arrays(splits, ytrue_resid, ypred_resid):
    """Align residual arrays with value rows (last index of each window) for plotting."""
    # We’ll extract columns from 'val' split to a numpy array and cut to min length.
    val = splits["val"].copy()
    # Build total PM (physics + residual true/pred) for PM10 as example
    # We need W_spec and PF to get physics baseline; let caller ensure it's present.
    return val

# ---------------------------
# FIGURE 5 – Physics vs Hybrid: PM10 vs W_spec (scatter + baseline line)
# ---------------------------

def plot_physics_vs_hybrid(splits, ytrue_resid, ypred_resid, pf, fname="figures/fig5_physics_vs_hybrid_pm10.png"):
    """
    Scatter:
      X = W_spec
      Y_true_total = PM10_true ≈ PF_PM10 * W_spec + residual_true_PM10
      Y_pred_total = PF_PM10 * W_spec + residual_pred_PM10
    Baseline: y = PF_PM10 * x (line)
    """
    # Ensure physics columns on val
    val = _augment_with_physics(splits["val"].copy(), pf=pf)
    # Align lengths with predictions (one y per window)
    # Simplest robust alignment: truncate both to min len
    m = min(len(val), ytrue_resid.shape[0], ypred_resid.shape[0])
    if m == 0:
        print("[plot] no data to plot physics vs hybrid.")
        return
    val = val.iloc[:m].reset_index(drop=True)
    W = val["W_spec"].to_numpy()[:m]
    # find index of PM10 in columns
    pm_index = cfg.target_cols.index("PM10_ug_per_m3")
    PF = pf["PM10"]

    # totals
    y_true_total = PF * W + ytrue_resid[:m, pm_index]
    y_pred_total = PF * W + ypred_resid[:m, pm_index]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(W, y_true_total, s=8, alpha=0.5, label="Measured PM10")
    ax.scatter(W, y_pred_total, s=8, alpha=0.5, label="Hybrid PM̂10")
    # baseline line
    xs = np.linspace(W.min(), W.max(), 100)
    ax.plot(xs, PF * xs, lw=2, label="Physics baseline: PF · W_spec")

    ax.set_xlabel("Specific friction work $W_{spec}$ [J/kg]")
    ax.set_ylabel("PM10 [$\\mu g/m^3$]")
    ax.legend()
    ax.grid(True, alpha=0.4)
    Path("figures").mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {fname}")

# ---------------------------
# FIGURE 6 – Driving strategy map (speed vs accel → mean PM10)
# ---------------------------

def plot_strategy_map(splits, ypred_resid, pf, fname="figures/fig6_strategy_map_pm10.png"):
    """
    Heatmap of driving strategy → PM10:
      X = speed_kmh
      Y = acc_long_ms2
      Color = mean total PM10 (PF*W_spec + predicted residual)
    """
    val = _augment_with_physics(splits["val"].copy(), pf=pf)
    # Align to predictions
    m = min(len(val), ypred_resid.shape[0])
    if m == 0:
        print("[plot] no data to plot strategy map.")
        return
    val = val.iloc[:m].reset_index(drop=True)
    W = val["W_spec"].to_numpy()[:m]
    pm_index = cfg.target_cols.index("PM10_ug_per_m3")
    PF = pf["PM10"]
    pm_total_pred = PF * W + ypred_resid[:m, pm_index]

    spd = val["speed_kmh"].to_numpy()[:m]
    axx = val["acc_long_ms2"].to_numpy()[:m]

    # 2D binning (official matplotlib hexbin/hist2d patterns)
    fig, ax = plt.subplots(figsize=(7, 5))
    h = ax.hist2d(spd, axx, weights=pm_total_pred, bins=50, cmap="viridis")
    # Convert to mean by dividing by counts:
    counts, _, _ = np.histogram2d(spd, axx, bins=50)
    # h[0] currently holds sum(weights) per bin; divide by counts -> mean
    z = h[0]
    # Avoid div by zero
    counts[counts == 0] = 1
    mean_pm = z / counts
    ax.clear()
    im = ax.imshow(mean_pm.T, origin="lower", aspect="auto",
                   extent=[spd.min(), spd.max(), axx.min(), axx.max()],
                   cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean PM10 [$\\mu g/m^3$]")
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("Longitudinal accel $a_x$ [m/s$^2$]")
    ax.set_title("Driving strategy map → PM10")

    Path("figures").mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {fname}")

# ---------------------------
# RUN ALL FIGURES
# ---------------------------

def main():
    Path("figures").mkdir(parents=True, exist_ok=True)

    # 1) Load & preprocess exactly as in training
    df = pd.read_csv(cfg.data_path).dropna(subset=cfg.target_cols)
    splits = prepare_frames(df)

    # 2) Fit physics PF on TRAIN only (Feißel-style baseline)
    pf = fit_pf_per_target(splits["train"], cfg)  # keys: "PM10", "PM2_5", "PM1"

    # 3) Build model & validation loader (must mirror train.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # same feature/target sets as training
    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    out_dim = len(targ_cols)
    model = _build_model(
        device=device,
        out_dim=out_dim,
        input_size=len(feat_cols),
    )

    val_loader = _make_val_loader(splits)
    ytrue_resid, ypred_resid = _predict_loader(model, val_loader, device)
    if ytrue_resid is None:
        print("[report] no validation windows; nothing to plot.")
        return

    # ---- FIGURE 1
    plot_pipeline_diagram("figures/fig1_pipeline.png")

    # ---- FIGURE 4
    plot_val_overlay_from_loader(model, val_loader, cfg.target_cols,
                                 "figures/fig4_overlay_val_residuals.png")

    # ---- FIGURE 5
    plot_physics_vs_hybrid(splits, ytrue_resid, ypred_resid, pf,
                           "figures/fig5_physics_vs_hybrid_pm10.png")

    # ---- FIGURE 6
    plot_strategy_map(splits, ypred_resid, pf,
                      "figures/fig6_strategy_map_pm10.png")

    print("[report] all plots generated. See 'figures/'.")

if __name__ == "__main__":
    main()
