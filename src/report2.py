# src/report2.py
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Make matplotlib/headless safe on WSL/servers
os.environ.setdefault("MPLBACKEND", "Agg")

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
from src.models import RNNRegressor, TCNRegressor


# Extra plots/utilities you created in report_more.py
from src.report_more import (
    plot_hybrid_vs_physics,        # (aligned_df, pred_df, out_png)
    binned_mae,                    # (y_true, y_pred, bins, out_png)
    brake_onset_average,           # (aligned_df, pred_df, pre_s, post_s, hz) -> DataFrame
    plot_brake_onset_average,      # (avg_df, out_png)
    residual_autocorr,             # (residuals, out_png)
    rolling_mae,                   # (y_true, y_pred, window, out_png)
    permutation_feature_importance # (model, loader, feat_names, out_png)
)

def _prep_splits():
    """
    Use the same pipeline as training: prepare_frames() builds residual/physics
    targets and returns {'train','val','test'} splits already transformed.
    """
    df = pd.read_csv(cfg.data_path)
    return prepare_frames(df)

@torch.no_grad()
def _predict_df(model, ds):
    """
    Return DataFrame with columns: PM10_true/pred, PM2_5_true/pred, PM1_true/pred
    (predictions are in residual space, matching training targets).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    truths, preds = [], []
    device = next(model.parameters()).device

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

    # Map to base names for plots
    bases = [c.split("_ug")[0] if "_ug" in c else c for c in cfg.target_cols]

    out = {}
    for i, b in enumerate(bases):
        out[f"{b}_true"] = ytrue[:, i]
        out[f"{b}_pred"] = ypred[:, i]
    return pd.DataFrame(out)

def main():
    Path("figures").mkdir(parents=True, exist_ok=True)

    # 1) Prepare data (identical logic to training/evaluate)
    splits = _prep_splits()
    test_df = splits["test"].reset_index(drop=True)

    # Windowing must match training
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    feat_cols = resolved_feature_cols(cfg)          # features used in training
    targ_cols = resolved_target_cols(cfg)           # residual (or physics residual) targets used in training

    test_ds = WindowedDataset(test_df, feat_cols, targ_cols, window, stride)

        # 2) Load model (match training config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt = cfg.model_type.lower()

    if mt in ("lstm", "gru"):
        model = RNNRegressor(
            input_size=len(feat_cols),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.model_type,          # "lstm" or "gru"
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=len(targ_cols),
        ).to(device)
    elif mt == "tcn":
        model = TCNRegressor(
            input_size=len(feat_cols),
            out_dim=len(targ_cols),
            channels_list=getattr(cfg, "tcn_channels", (64, 64, 64)),
            kernel_size=getattr(cfg, "tcn_kernel_size", 3),
            dropout=cfg.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown cfg.model_type={cfg.model_type}")

    ckpt = f".runs/{cfg.model_type}_best.pt"   # e.g. .runs/tcn_best.pt
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 3) Predictions in residual space
    pred_df = _predict_df(model, test_ds)
    if pred_df.empty:
        print("[report2] No test windows; nothing to plot.")
        return

    # determine target dynamically
    t0 = cfg.target_cols[0]
    base0 = t0.split("_ug")[0] if "_ug" in t0 else t0

    # Align regime/aux columns with end-of-window targets
    end_idxs = [e - 1 for (s, e) in test_ds.indices]
    aligned = test_df.iloc[end_idxs].reset_index(drop=True)



    # 3a) Binned MAE vs speed
    binned_mae(
        pred_df[f"{base0}_true"].values,
        pred_df[f"{base0}_pred"].values,
        aligned["speed_kmh"].values,
        20,
        f"figures/fig_binned_mae_{base0.lower()}_speed.png",
    )

    # 3b) Residual autocorrelation
    residual_autocorr(
        pred_df[f"{base0}_true"].values - pred_df[f"{base0}_pred"].values,
        f"figures/fig_acf_{base0.lower()}_resid.png",
    )

    # 3c) Rolling MAE
    rolling_mae(
        pred_df[f"{base0}_true"].values,
        pred_df[f"{base0}_pred"].values,
        3000,
        f"figures/fig_rolling_mae_{base0.lower()}.png",
    )

    # 3d) Brake-onset average residuals
    if base0 == "PM2_5":
        avg_df = brake_onset_average(aligned, pred_df, 5, 15, cfg.target_hz)
        plot_brake_onset_average(avg_df, "figures/fig_brake_onset_avg_pm25.png")
    else:
        print(f"[report2] skipped brake-onset plot (target={base0})")

    # 3e) Hybrid vs physics overlay for PM10 (if physics cols exist)
    if base0.startswith("PM"):
        try:
            plot_hybrid_vs_physics(
                aligned,
                pred_df,
                f"figures/fig_hybrid_vs_physics_{base0.lower()}.png",
                target_col=t0,
            )
        except Exception as e:
            print("[report2] Skipped hybrid_vs_physics:", e)
    else:
        print(f"[report2] skipped hybrid_vs_physics (target={base0})")

    # 3f) Permutation feature importance (small loader; may be slow)
    try:
        from torch.utils.data import DataLoader
        loader_small = DataLoader(test_ds, batch_size=min(64, cfg.batch_size), shuffle=False)
        permutation_feature_importance(
            model,
            loader_small,
            feat_cols,
            "figures/fig_permutation_importance.png",
        )
    except Exception as e:
        print("[report2] Skipped permutation importance:", e)

    print("[report2] wrote extra figs in 'figures/'")

if __name__ == "__main__":
    main()
