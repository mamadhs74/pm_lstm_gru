# src/report_more.py
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# -----------------------------
# 1) BINNED MAE vs a feature
# -----------------------------
def binned_mae(y_true, y_pred, by, nbins, out_png):
    """
    Compute MAE in bins of a conditioning variable `by` (e.g., brake_bar),
    then plot MAE vs bin center.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "by": by})
    # robust bin edges: fall back to unique-only if distribution is degenerate
    q = np.linspace(0, 1, nbins + 1)
    edges = np.unique(np.quantile(df["by"].to_numpy(float), q))
    if edges.size < 2:  # all identical
        edges = np.array([df["by"].min() - 1e-9, df["by"].max() + 1e-9])

    df["bin"] = pd.cut(df["by"], bins=edges, include_lowest=True, duplicates="drop")
    grp = df.groupby("bin", observed=True)

    # compute MAE per bin without triggering FutureWarning
    mae = (
        grp[["y_true", "y_pred"]]
        .apply(lambda g: np.mean(np.abs(g["y_true"] - g["y_pred"])))
        .rename("MAE")
        .reset_index()
    )

    centers = []
    for iv in mae["bin"]:
        # iv is a pandas.Interval
        centers.append((iv.left + iv.right) / 2.0)
    mae["center"] = centers

    plt.figure()
    plt.plot(mae["center"].to_numpy(float), mae["MAE"].to_numpy(float), marker="o")
    plt.xlabel("Binning variable")
    plt.ylabel("MAE")
    plt.title("Binned MAE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------------------
# 2) Residual autocorrelation (simple ACF)
# ----------------------------------------
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

def ljung_box_p(residuals: np.ndarray, lags: int = 40) -> float:
    # returns p-value at the max lag
    r = residuals.astype(float)
    r = r - np.nanmean(r)
    lb = acorr_ljungbox(r, lags=[lags], return_df=True)  # needs statsmodels
    return float(lb["lb_pvalue"].iloc[0])

def residual_autocorr(residuals, out_png, max_lag=200, show_ci=True, title="Residual Autocorrelation"):
    """
    Publication-grade residual ACF:
      - mean-centered
      - standard variance normalization
      - optional 95% white-noise CI bands (~ ±1.96/sqrt(N))
    """
    import numpy as np
    import matplotlib.pyplot as plt

    r = np.asarray(residuals, dtype=float).ravel()
    if r.size < 3:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "Not enough data for ACF", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return

    # Handle NaNs conservatively
    m = np.nanmean(r)
    r = np.nan_to_num(r - m, nan=0.0)

    n = r.size
    max_lag = int(min(max_lag, n - 2))
    if max_lag < 1:
        max_lag = 1

    denom = float(np.dot(r, r))
    if denom <= 0.0:
        denom = 1.0

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for k in range(1, max_lag + 1):
        acf[k] = float(np.dot(r[:-k], r[k:]) / denom)

    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem(lags, acf)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if show_ci:
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, linewidth=1.0, linestyle="--")
        ax.axhline(-ci, linewidth=1.0, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_error_acf_panel(aligned_df, pred_df, out_png, target_short="PM2_5", max_lag=200):
    """
    Creates an ACF comparison:
      - physics-only error: y - y_phys
      - hybrid error:       y - (y_phys + r_hat)
    Requires aligned_df to be aligned to window END indices.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    phys_col = f"{target_short}_phys"
    resid_pred_col = f"{target_short}_pred"

    # Truth in physical units:
    # if you store original target column names, adapt here
    # (common case in your codebase: PM2_5_ug_per_m3)
    truth_candidates = [f"{target_short}_ug_per_m3", target_short]
    truth_col = next((c for c in truth_candidates if c in aligned_df.columns), None)
    if truth_col is None:
        raise ValueError(f"Could not find truth column for {target_short}. Tried {truth_candidates}")

    if phys_col not in aligned_df.columns:
        raise ValueError(f"Missing {phys_col} in aligned_df")
    if resid_pred_col not in pred_df.columns:
        raise ValueError(f"Missing {resid_pred_col} in pred_df")

    y = aligned_df[truth_col].to_numpy(dtype=float)
    y_phys = aligned_df[phys_col].to_numpy(dtype=float)
    r_hat = pred_df[resid_pred_col].to_numpy(dtype=float)

    n = min(len(y), len(y_phys), len(r_hat))
    y = y[:n]
    y_phys = y_phys[:n]
    r_hat = r_hat[:n]

    e_phys = y - y_phys
    e_hyb = y - (y_phys + r_hat)

    # Compute ACF with the same estimator as residual_autocorr()
    def _acf(r, max_lag):
        r = r - np.nanmean(r)
        r = np.nan_to_num(r, nan=0.0)
        denom = float(np.dot(r, r)) or 1.0
        L = int(min(max_lag, len(r) - 2))
        out = np.empty(L + 1, dtype=float)
        out[0] = 1.0
        for k in range(1, L + 1):
            out[k] = float(np.dot(r[:-k], r[k:]) / denom)
        return out

    a1 = _acf(e_phys, max_lag)
    a2 = _acf(e_hyb, max_lag)
    L = min(len(a1), len(a2)) - 1
    lags = np.arange(L + 1)

    ci = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, a1[:L+1], label="ACF: physics-only error")
    ax.plot(lags, a2[:L+1], label="ACF: hybrid error")
    ax.axhline(ci, linewidth=1.0, linestyle="--")
    ax.axhline(-ci, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Error ACF comparison — {target_short}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# 3) Rolling MAE over time
# -----------------------------
def rolling_mae(y_true, y_pred, window, out_png):
    """
    Rolling mean absolute error with a given window length (samples).
    """
    y_true = pd.Series(np.asarray(y_true).ravel())
    y_pred = pd.Series(np.asarray(y_pred).ravel())
    mae = (y_true - y_pred).abs()
    roll = mae.rolling(window=window, min_periods=1, center=False).mean()

    plt.figure(figsize=(7, 4))
    plt.plot(roll.values)
    plt.xlabel("Sample")
    plt.ylabel(f"Rolling MAE (window={window})")
    plt.title("Rolling MAE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------------------------------
# 4) Brake-onset average residuals (event-triggered)
# ----------------------------------------------------
def _find_brake_onsets(brake_series):
    """Indices where brake_event rises 0 -> 1."""
    b = np.asarray(brake_series).astype(int)
    prev = np.roll(b, 1)
    prev[0] = 0
    onsets = np.where((prev == 0) & (b == 1))[0]
    return onsets

def brake_onset_average(aligned_df, pred_df, pre_s, post_s, hz):
    """
    Compute average PM2.5 residual around brake onset events.
    Returns a DataFrame with time axis and mean residual.
    """
    resid = pred_df["PM2_5_true"].values - pred_df["PM2_5_pred"].values
    b = aligned_df["brake"] if "brake" in aligned_df.columns else aligned_df.get("brake_event", 0)
    onsets = _find_brake_onsets(b)

    pre_n = int(pre_s * hz)
    post_n = int(post_s * hz)
    win_len = pre_n + post_n + 1

    snippets = []
    for idx in onsets:
        start = idx - pre_n
        end = idx + post_n + 1
        if start < 0 or end > len(resid):
            continue
        snippets.append(resid[start:end])

    if not snippets:
        # Return empty but well-formed
        return pd.DataFrame({"t_s": np.arange(-pre_n, post_n + 1) / hz, "mean_resid": np.nan})

    M = np.vstack(snippets)  # [n_events, win_len]
    mean = np.nanmean(M, axis=0)
    t_s = np.arange(-pre_n, post_n + 1) / float(hz)
    return pd.DataFrame({"t_s": t_s, "mean_resid": mean})

def plot_brake_onset_average(avg_df, out_png):
    plt.figure(figsize=(6, 4))
    plt.plot(avg_df["t_s"], avg_df["mean_resid"])
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time since brake onset (s)")
    plt.ylabel("Mean residual (PM2.5)")
    plt.title("Brake-Onset Average Residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -------------------------------------------------------
# 5) Hybrid (physics + residual) vs physics-only overlay
# -------------------------------------------------------
def plot_hybrid_vs_physics(aligned_df, pred_df, out_png, target_col="PM10_ug_per_m3"):
    """
    If 'aligned_df' contains physics baseline columns (e.g., 'PM10_phys'),
    reconstruct hybrid = physics + residual_pred and plot vs truth.
    """
    # Short name mapping from your pipeline
    short = target_col.split("_ug")[0]  # "PM10_ug_per_m3" -> "PM10"
    phys_col = f"{short}_phys"

    if phys_col not in aligned_df.columns:
        raise ValueError(f"Missing physics baseline column '{phys_col}' in aligned_df.")

    # Physics baseline
    y_phys = aligned_df[phys_col].to_numpy()

    # Residual prediction (use the correct column from pred_df)
    resid_pred_col = f"{short}_pred"
    if resid_pred_col not in pred_df.columns:
        raise ValueError(f"Missing residual prediction column '{resid_pred_col}' in pred_df.")

    y_resid_pred = pred_df[resid_pred_col].to_numpy()

    # Hybrid = physics + residual_pred
    y_hybrid = y_phys[: len(y_resid_pred)] + y_resid_pred

    # Truth (original PM column)
    if target_col not in aligned_df.columns:
        raise ValueError(f"Missing ground-truth target '{target_col}' in aligned_df.")
    y_true = aligned_df[target_col].to_numpy()[: len(y_resid_pred)]

    # Plot a limited window for readability (e.g., first 2000 samples)
    N = min(len(y_true), 2000)
    t = np.arange(N)

    plt.figure(figsize=(9, 4))
    plt.plot(t, y_true[:N], label="Truth", linewidth=1)
    plt.plot(t, y_phys[:N], label="Physics-only", linewidth=1)
    plt.plot(t, y_hybrid[:N], label="Hybrid (phys + residual)", linewidth=1)
    plt.legend()
    plt.xlabel("Sample")
    plt.ylabel(short)  # human-readable
    plt.title(f"Hybrid vs Physics — {short}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------------------------------------------------------
# 6) Very simple permutation feature importance (sequence)
# ---------------------------------------------------------
@torch.no_grad()
def permutation_feature_importance(model, loader, feat_names, out_png, max_batches=3):
    """
    Sequence-aware permutation importance:
      - Compute baseline MAE on a few batches.
      - For each feature f, permute that column across the batch (keeps T) and
        recompute MAE; report delta MAE.

    Assumes loader yields (xb, yb) with xb shape [B, T, F].
    """
    device = next(model.parameters()).device
    model.eval()

    # Helper to compute MAE over loader (few batches)
    def _mae_over(loader):
        errs = []
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yp = model(xb)
            if isinstance(yp, (tuple, list)):
                yp = yp[0]

            errs.append(torch.mean(torch.abs(yp - yb)).item())
            count += 1
            if count >= max_batches:
                break
        return float(np.mean(errs)) if errs else np.nan

    # Snapshot a small list of (xb, yb) for reuse
    small = []
    for i, (xb, yb) in enumerate(loader):
        small.append((xb.clone(), yb.clone()))
        if i + 1 >= max_batches:
            break

    # Baseline
    base_err = _mae_over(small)

    # Per-feature delta
    deltas = []
    for f_idx, name in enumerate(feat_names):
        # Copy minibatches and permute feature f across the *batch* axis
        pert_errs = []
        for xb, yb in small:
            xb = xb.clone().to(device)
            yb = yb.to(device)

            B, T, F = xb.shape
            # Permute feature f_idx independently across batch, for each time step
            perm = torch.randperm(B, device=xb.device)
            xb[:, :, f_idx] = xb[perm, :, f_idx]

            yp = model(xb)
            pert_errs.append(torch.mean(torch.abs(yp - yb)).item())

        deltas.append((name, float(np.mean(pert_errs) - base_err)))

    # Plot
    names = [n for n, _ in deltas]
    vals = [v for _, v in deltas]

    plt.figure(figsize=(max(6, 0.4 * len(names)), 4))
    plt.bar(names, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("ΔMAE vs baseline")
    plt.title("Permutation Feature Importance (lower is better)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
