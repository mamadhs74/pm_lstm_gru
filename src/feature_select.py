# src/feature_select.py
import argparse, os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rich import print

from .config import cfg
from .preprocess import downsample, add_regime_features, leak_free_split, fit_transformers, apply_transformers, build_residual_targets, shift_targets
from .datasets import WindowedDataset
from .models import RNNRegressor
from .metrics import mae, HuberLoss
from .attr import integrated_grads_feature_importance, plot_feature_importance

from torch.utils.data import DataLoader

def prepare_frames(df):
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    splits = leak_free_split(df, cfg)

    scaler = fit_transformers(splits["train"], cfg)  # fit on train only (official sklearn pattern)
    for k in splits:
        splits[k] = apply_transformers(splits[k], scaler, cfg)
        splits[k] = build_residual_targets(splits[k], cfg)   # online EMA (causal)
        splits[k] = shift_targets(splits[k], cfg)            # drop last τ to avoid peeking
    return splits

def make_loader(df):
    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps
    feat_cols = cfg.feature_cols + ["idle","brake_event","dv","jerk"]
    ds = WindowedDataset(df, feat_cols, cfg.target_cols, window, stride, regime_weights=None)
    return DataLoader(ds, batch_size=128, shuffle=False, drop_last=False), feat_cols

@torch.no_grad()
def baseline_val_mae(model, loader, device, criterion):
    model.eval()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yp = model(xb)
        losses.append(criterion(yp, yb).item())
    return float(np.mean(losses))

@torch.no_grad()
def permutation_importance(model, loader, feat_cols, device, criterion):
    """
    Model-agnostic permutation importance on the val set, implemented on the PyTorch DataLoader.
    We permute each feature across the batch dimension for each time step (keeps time structure inside a window).
    Returns: pd.Series of ΔMAE (permuted - baseline) per feature.
    """
    base = baseline_val_mae(model, loader, device, criterion)

    deltas = {f: [] for f in feat_cols}
    for f_idx, f_name in enumerate(feat_cols):
        # iterate once through val loader, compute MAE when feature f is permuted
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb_perm = xb.clone()
            # permute across batch at each time step for feature f_idx
            B, T, F = xb_perm.shape
            idx = torch.randperm(B, device=device)
            # For each time step, permute across batch (preserves each window's temporal structure)
            for t in range(T):
                xb_perm[:, t, f_idx] = xb_perm[idx, t, f_idx]
            yp = model(xb_perm)
            losses.append(criterion(yp, yb).item())
        deltas[f_name] = float(np.mean(losses) - base)

    s = pd.Series(deltas).sort_values(ascending=False)
    return base, s

def load_model(device):
    input_size = len(cfg.feature_cols) + 4
    model = RNNRegressor(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,            # "lstm" or "gru"
        conv_frontend=cfg.conv_frontend,
        conv_channels=cfg.conv_channels,
        out_dim=len(cfg.target_cols)
    ).to(device)
    ckpt = f".runs/{cfg.model_type}_best.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Train first: pixi run train")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=16, help="how many top features to suggest")
    parser.add_argument("--target_index", type=int, default=0, help="0: PM10, 1: PM2_5, 2: PM1 for IG ranking")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load + prepare frames (leak-free, per official practice)
    df = pd.read_csv(cfg.data_path)
    splits = prepare_frames(df)
    val_loader, feat_cols = make_loader(splits["val"])

    # 2) Load trained model
    model = load_model(device)

    # 3) Criterion (same as training)
    criterion = HuberLoss(cfg.huber_delta) if cfg.use_huber else mae

    # 4) Permutation importance (official sklearn method, adapted to DL val loader)
    base_mae, perm_series = permutation_importance(model, val_loader, feat_cols, device, criterion)
    print(f"[Permutation] baseline_val_MAE: {base_mae:.4f}")
    perm_series.to_csv("feature_importance_permutation.csv")
    # quick bar plot
    perm_series.head(args.topk).plot(kind="bar")
    plt.title("Permutation Importance (ΔMAE)")
    plt.tight_layout(); plt.savefig("figures/perm_importance.png", dpi=150, bbox_inches="tight"); plt.close()

    # 5) Integrated Gradients (official Captum)
    # gather a small batch from val for IG
    xb, _ = next(iter(val_loader))
    ig_series = integrated_grads_feature_importance(model, xb, feat_cols, target_index=args.target_index, steps=32, device=device)
    ig_series.to_csv("feature_importance_integrated_gradients.csv")
    plot_feature_importance(ig_series, "figures/ig_feature_importance.png", topk=args.topk)

    # 6) Combined suggestion: intersect top-k from both lists (consensus)
    top_perm = set(perm_series.head(args.topk).index.tolist())
    top_ig   = set(ig_series.head(args.topk).index.tolist())
    consensus = list(top_perm.intersection(top_ig))

    # note: we ALWAYS keep your engineered "idle/brake_event/dv/jerk" even if not in feature_cols
    # because they aren't in cfg.feature_cols; they are appended later anyway.

    with open("suggested_feature_cols.txt", "w", encoding="utf-8") as f:
        f.write("# Paste these into cfg.feature_cols in src/config.py\n")
        for name in consensus:
            # Only keep base/raw features in cfg.feature_cols (engineered ones are auto-added)
            if name in cfg.feature_cols:
                f.write(f"\"{name}\",\n")

    print("[Done] Wrote:")
    print(" - feature_importance_permutation.csv")
    print(" - feature_importance_integrated_gradients.csv")
    print(" - figures/perm_importance.png")
    print(" - figures/ig_feature_importance.png")
    print(" - suggested_feature_cols.txt  (paste into src/config.py)")
    print(f"[Tip] Run again with --target_index=1 or 2 to check PM2.5 / PM1.")
    
if __name__ == "__main__":
    main()
