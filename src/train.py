# src/train.py
# Training script following official PyTorch patterns:
# - model.train()/eval(), no_grad() for val
# - opt.zero_grad -> forward -> loss.backward -> clip_grad_norm_ -> opt.step
# - optional CUDA AMP for speed/stability
# - Cosine LR scheduler stepped once per epoch
# - TensorBoard logging (scalars + throttled gradient histograms)
# - Saves best checkpoint; saves figures and a quick val overlay

import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich import print
from torch.utils.tensorboard import SummaryWriter

from .grad_viz import make_writer, log_gradients_tb
from .config import cfg
from .preprocess import (
    downsample,
    add_regime_features,
    leak_free_split,
    fit_transformers,
    apply_transformers,
    build_residual_targets,
    shift_targets,
)
from .datasets import make_loaders, WindowedDataset
from .models import RNNRegressor
from .metrics import mae, HuberLoss
from .visualize import plot_overlay, plot_training_curves, plot_grad_norms


# --------------------------- utils ---------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Strong determinism (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning: drop rows missing targets
    return df.dropna(subset=cfg.target_cols)


def prepare_frames(df: pd.DataFrame):
    df = downsample(df, cfg.orig_hz, cfg.target_hz)
    df = add_regime_features(df)
    splits = leak_free_split(df, cfg)

    scaler = fit_transformers(splits["train"], cfg)
    for k in splits:
        splits[k] = apply_transformers(splits[k], scaler, cfg)
        splits[k] = build_residual_targets(splits[k], cfg)
        splits[k] = shift_targets(splits[k], cfg)
    return splits


@torch.no_grad()
def predict_loader_to_df(model: torch.nn.Module,
                         loader: torch.utils.data.DataLoader,
                         target_cols,
                         device):
    model.eval()
    ys_true, ys_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yp = model(xb)
        ys_true.append(yb.cpu().numpy())
        ys_pred.append(yp.cpu().numpy())
    if not ys_true:
        return pd.DataFrame()
    ytrue = np.vstack(ys_true)
    ypred = np.vstack(ys_pred)
    df_out = pd.DataFrame()
    for i, c in enumerate(target_cols):
        base = c.split("_ug")[0]
        df_out[f"{base}_true"] = ytrue[:, i]
        df_out[f"{base}_pred"] = ypred[:, i]
    return df_out


# --------------------------- training ---------------------------

def train_one_split(train_df: pd.DataFrame, val_df: pd.DataFrame, device: torch.device):
    os.makedirs("figures", exist_ok=True)
    os.makedirs("runs", exist_ok=True)    # re-create if deleted
    os.makedirs(".runs", exist_ok=True)

    # DataLoaders
    train_loader, val_loader = make_loaders(train_df, val_df, cfg)

    # Hard checks: if no windows, fail loudly with a helpful message
    n_train_win = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
    n_val_win   = len(val_loader.dataset) if hasattr(val_loader, "dataset") else 0
    print(f"[debug] windows -> train: {n_train_win}, val: {n_val_win}")
    if n_train_win == 0 or n_val_win == 0:
        raise RuntimeError(
            f"No windows to train/validate. Got train={n_train_win}, val={n_val_win}. "
            f"Reduce cfg.window_s / cfg.target_shift_s, or ensure your CSV is long enough."
        )

    # Model
    input_size = len(cfg.feature_cols) + 4  # + idle, brake_event, dv, jerk
    model = RNNRegressor(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,
        conv_frontend=cfg.conv_frontend,
        conv_channels=cfg.conv_channels,
        out_dim=len(cfg.target_cols),
    ).to(device)

    run_name = f"{cfg.model_type}_hz{cfg.target_hz}_win{cfg.window_s}"
    writer: SummaryWriter = make_writer(logdir="runs", run_name=run_name)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    criterion = HuberLoss(cfg.huber_delta) if cfg.use_huber else mae

    best_val = float("inf")
    best_state = None
    global_step = 0

    epoch_train_losses = []
    epoch_val_losses   = []
    grad_norms_per_epoch = []

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("[info] starting training loop")
    try:
        for epoch in range(cfg.max_epochs):
            model.train()
            batch_losses = []
            batch_grad_norms = []

            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(xb)
                    loss = criterion(pred, yb)

                scaler.scale(loss).backward()
                # Clip after unscale (official AMP pattern)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(opt)
                scaler.update()

                # Throttle gradient histograms to avoid TB spam
                if global_step % 50 == 0:
                    log_gradients_tb(writer, model, step=global_step)

                # Global grad norm (for our per-epoch plot)
                total_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().flatten()
                        total_sq += float(torch.dot(g, g))
                batch_grad_norms.append(np.sqrt(total_sq))

                batch_losses.append(loss.item())
                global_step += 1

            # Validation
            model.eval()
            vloss = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    vloss.append(criterion(pred, yb).item())

            mtrain = float(np.mean(batch_losses)) if batch_losses else float("nan")
            mval   = float(np.mean(vloss))        if vloss        else float("nan")
            epoch_train_losses.append(mtrain)
            epoch_val_losses.append(mval)
            grad_norms_per_epoch.append(
                float(np.mean(batch_grad_norms)) if batch_grad_norms else float("nan")
            )

            writer.add_scalar("loss/train", mtrain, epoch)
            writer.add_scalar("loss/val",   mval,   epoch)
            print(f"[epoch {epoch:03d}] train {mtrain:.4f}  val {mval:.4f}", flush=True)

            sch.step()

            if mval < best_val:
                best_val = mval
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    finally:
        # Save curves even if something fails mid-run
        plot_training_curves(epoch_train_losses, epoch_val_losses, "figures/training_curves.png")
        plot_grad_norms(grad_norms_per_epoch, "figures/grad_norms.png")
        writer.close()

    return {"val": best_val, "state": best_state}


# --------------------------- main ---------------------------

def main():
    set_seed(cfg.seed)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs(".runs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" device: {device}")

    # Load & prep
    df = load_csv(cfg.data_path)
    print(f" loaded {len(df)} rows from {cfg.data_path}")
    splits = prepare_frames(df)

    # Dataset/window sizes (helpful debug)
    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps
    feat_cols = cfg.feature_cols + ["idle", "brake_event", "dv", "jerk"]
    counts = {}
    for name in ["train", "val", "test"]:
        ds = WindowedDataset(splits[name], feat_cols, cfg.target_cols, window, stride)
        counts[name] = (len(splits[name]), len(ds))
    print(f" rows/windows -> {counts}")

    # Train
    best = train_one_split(splits["train"], splits["val"], device)

    # Save best
    ckpt = f".runs/{cfg.model_type}_best.pt"
    torch.save(best["state"], ckpt)
    print(f" saved best model to {ckpt}")

    # Quick val overlay (residual targets)
    from torch.utils.data import DataLoader
    pin = torch.cuda.is_available()
    val_ds = WindowedDataset(splits["val"], feat_cols, cfg.target_cols, window, stride)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=pin,
    )

    model = RNNRegressor(
        input_size=len(cfg.feature_cols) + 4,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,
        conv_frontend=cfg.conv_frontend,
        conv_channels=cfg.conv_channels,
        out_dim=len(cfg.target_cols),
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    df_pred = predict_loader_to_df(model, val_loader, cfg.target_cols, device)
    if not df_pred.empty:
        out_csv = "data/processed/val_predictions_residuals.csv"
        df_pred.to_csv(out_csv, index=False)
        plot_overlay(df_pred, "figures/overlay_val_residuals.png",
                     title="Validation — residuals (true vs pred)")
        print(" wrote:")
        print("  - figures/training_curves.png")
        print("  - figures/grad_norms.png")
        print("  - figures/overlay_val_residuals.png")
        print(f"  - {out_csv}")
    else:
        print(" [warn] validation predictions empty (no windows).")


if __name__ == "__main__":
    main()
