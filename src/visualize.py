# src/visualize.py
# Matplotlib visualizations using the official, object-oriented API.
# - No seaborn, no global style changes
# - Each function creates its own Figure/Axes and saves with tight_layout + bbox_inches="tight"

from __future__ import annotations
import os
from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- utils ----------

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ---------- core time-series overlay ----------

def plot_overlay(
    df: pd.DataFrame,
    fname: str,
    time_col: Optional[str] = None,
    title: str = "PM predictions vs truth",
    targets: Sequence[str] = ("PM10", "PM2_5", "PM1"),
) -> None:
    """
    Plots true vs predicted series for each target on the same axes.
    Expects columns like: PM10_true, PM10_pred, PM2_5_true, PM2_5_pred, PM1_true, PM1_pred.
    """
    _ensure_dir(fname)

    # x-axis selection
    x = df[time_col].values if (time_col and time_col in df.columns) else df.index.values

    fig, ax = plt.subplots(figsize=(10, 4))
    for t in targets:
        true_col = f"{t}_true"
        pred_col = f"{t}_pred"
        if true_col in df and pred_col in df:
            ax.plot(x, df[true_col].values, label=f"{t} true")
            ax.plot(x, df[pred_col].values, linestyle="--", alpha=0.85, label=f"{t} pred")

    ax.set_title(title)
    ax.set_xlabel("time" if time_col else "sample index")
    ax.set_ylabel("µg/m³")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(ncol=min(3, len(targets) * 2))

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- training diagnostics ----------

def plot_training_curves(
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    fname: str = "figures/training_curves.png",
    title: str = "Loss per epoch",
    ylabel: str = "loss",
) -> None:
    """Plots train/val loss vs epoch."""
    _ensure_dir(fname)
    train_losses = list(train_losses)
    val_losses = list(val_losses)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grad_norms(
    grad_norms: Iterable[float],
    fname: str = "figures/grad_norms.png",
    title: str = "Gradient norm per epoch",
) -> None:
    """Plots global gradient norm vs epoch (aggregated per epoch)."""
    _ensure_dir(fname)
    grad_norms = list(grad_norms)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grad_norms, label="global grad norm")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("||∇||")
    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- prediction diagnostics ----------

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fname: str = "figures/parity.png",
    title: str = "Parity plot (pred vs true)",
    xlabel: str = "true",
    ylabel: str = "pred",
) -> None:
    """
    Parity (scatter) plot with y=x reference line.
    y_true/y_pred: arrays of shape (N,) or (N,1) or flattened beforehand.
    """
    _ensure_dir(fname)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_true, y_pred, s=8, alpha=0.6)
    # y=x reference
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lo, hi], [lo, hi], linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_hist(
    residuals: np.ndarray,
    fname: str = "figures/residual_hist.png",
    bins: int = 50,
    title: str = "Residuals histogram",
    xlabel: str = "residual (true - pred)",
) -> None:
    """Histogram of residuals to inspect bias/heavy tails."""
    _ensure_dir(fname)
    residuals = np.asarray(residuals).ravel()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.4, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- feature importance (e.g., Integrated Gradients) ----------

def plot_feature_importance_bar(
    series: pd.Series,
    fname: str = "figures/feature_importance.png",
    topk: int = 20,
    title: str = "Feature importance (top-k)"
) -> None:
    """
    Generic bar plot for importance series (e.g., Captum IG results).
    Expects a pandas Series indexed by feature name with numeric importance values.
    """
    _ensure_dir(fname)
    top = series.head(topk)

    fig, ax = plt.subplots(figsize=(10, 4))
    top.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("importance (arb. units)")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
