# src/train.py
import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich import print
from torch.utils.tensorboard import SummaryWriter

from .grad_viz import make_writer, log_gradients_tb
from .config import cfg, resolved_feature_cols, resolved_target_cols
from .preprocess import prepare_frames
from .datasets import make_loaders, WindowedDataset
from .models import RNNRegressor
from .metrics import mae, HuberLoss
from .visualize import plot_overlay, plot_training_curves, plot_grad_norms

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna(subset=cfg.target_cols)

@torch.no_grad()
def predict_loader_to_df(model, loader, display_target_cols, device):
    model.eval()
    ys_true, ys_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yp = model(xb)
        ys_true.append(yb.cpu().numpy())
        ys_pred.append(yp.cpu().numpy())
    if not ys_true: return pd.DataFrame()
    ytrue, ypred = np.vstack(ys_true), np.vstack(ys_pred)
    out = pd.DataFrame()
    for i, c in enumerate(display_target_cols):
        base = c.split("_ug")[0]
        out[f"{base}_true"] = ytrue[:, i]
        out[f"{base}_pred"] = ypred[:, i]
    return out

def train_one_split(train_df, val_df, device):
    os.makedirs("figures", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs(".runs", exist_ok=True)

    train_loader, val_loader = make_loaders(train_df, val_df, cfg)
    n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
    n_val   = len(val_loader.dataset)   if hasattr(val_loader,   "dataset") else 0
    print(f"[debug] windows -> train: {n_train}, val: {n_val}")
    if n_train == 0 or n_val == 0:
        raise RuntimeError("No windows in train/val. Adjust window/shift/CSV length.")

    # Model
    feats = resolved_feature_cols(cfg)
    targs = resolved_target_cols(cfg)
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

    run_name = f"{cfg.model_type}_hz{cfg.target_hz}_win{cfg.window_s}"
    writer: SummaryWriter = make_writer(logdir="runs", run_name=run_name)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    criterion = HuberLoss(cfg.huber_delta) if cfg.use_huber else mae

    best_val = float("inf")
    best_state = None
    global_step = 0
    epoch_train_losses, epoch_val_losses, grad_norms = [], [], []

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(" starting training loop")
    try:
        for epoch in range(cfg.max_epochs):
            model.train()
            batch_losses, batch_grad_norms = [], []

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = model(xb)
                        loss = criterion(pred, yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                if global_step % 50 == 0:
                    log_gradients_tb(writer, model, step=global_step)

                total_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().flatten()
                        total_sq += float(torch.dot(g, g))
                batch_grad_norms.append(np.sqrt(total_sq))
                batch_losses.append(float(loss.detach().cpu()))
                global_step += 1

            # ---- validation ----
            model.eval()
            vloss = []
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            vloss.append(criterion(model(xb), yb).item())
                else:
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        vloss.append(criterion(model(xb), yb).item())

            # ---- compute & log epoch metrics, step LR, track best ----
            mtrain = float(np.mean(batch_losses)) if batch_losses else float("nan")
            mval   = float(np.mean(vloss))        if vloss        else float("nan")
            epoch_train_losses.append(mtrain)
            epoch_val_losses.append(mval)
            grad_norms.append(float(np.mean(batch_grad_norms)) if batch_grad_norms else float("nan"))

            writer.add_scalar("loss/train", mtrain, epoch)
            writer.add_scalar("loss/val",   mval,   epoch)
            print(f" epoch {epoch:03d} | train {mtrain:.4f} | val {mval:.4f}")

            sch.step()
            if mval < best_val:
                best_val = mval
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}


    finally:
        plot_training_curves(epoch_train_losses, epoch_val_losses, "figures/training_curves.png")
        plot_grad_norms(grad_norms, "figures/grad_norms.png")
        writer.close()

    return {"val": best_val, "state": best_state}

def main():
    set_seed(cfg.seed)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs(".runs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" device: {device}")

    df = load_csv(cfg.data_path)
    print(f" loaded {len(df)} rows from {cfg.data_path}")
    splits = prepare_frames(df)

    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps
    feats = resolved_feature_cols(cfg)
    targs = resolved_target_cols(cfg)
    counts = {}
    for name in ["train", "val", "test"]:
        ds = WindowedDataset(splits[name], feats, targs, window, stride)
        counts[name] = (len(splits[name]), len(ds))
    print(f" rows/windows -> {counts}")

    best = train_one_split(splits["train"], splits["val"], device)

    ckpt = f".runs/{cfg.model_type}_best.pt"
    torch.save(best["state"], ckpt)
    print(f" saved best model to {ckpt}")

    # quick val overlay (residual targets)
    from torch.utils.data import DataLoader
    pin = torch.cuda.is_available()
    val_ds = WindowedDataset(splits["val"], feats, targs, window, stride)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=False, num_workers=2, pin_memory=pin)

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
    model.load_state_dict(torch.load(ckpt, map_location=device))
        # Safety: the human-friendly labels (cfg.target_cols) must match the model output order/length
    assert len(targs) == len(cfg.target_cols), "Resolved targets and display names must have same length."

    df_pred = predict_loader_to_df(model, val_loader, cfg.target_cols, device)
    if not df_pred.empty:
        out_csv = "data/processed/val_predictions_residuals.csv"
        df_pred.to_csv(out_csv, index=False)
        plot_overlay(df_pred, "figures/overlay_val_residuals.png",
                     title="Validation — residuals (true vs pred)")
        print(" wrote:\n  - figures/training_curves.png\n  - figures/grad_norms.png\n"
              "  - figures/overlay_val_residuals.png\n"
              f"  - {out_csv}")
    else:
        print(" [warn] validation predictions empty (no windows).")

if __name__ == "__main__":
    main()
