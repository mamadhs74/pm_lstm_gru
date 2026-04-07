# src/train.py
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich import print

from .grad_viz import make_writer, log_gradients_tb
from .config import cfg, resolved_feature_cols, resolved_target_cols
from .preprocess import prepare_frames
from .datasets import make_loaders, WindowedDataset
from .models import RNNRegressor, TCNRegressor
from .metrics import mae, HuberLoss
from .visualize import plot_training_curves, plot_grad_norms


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        if isinstance(yp, (tuple, list)):
            yp = yp[0]
        ys_true.append(yb.cpu().numpy())
        ys_pred.append(yp.cpu().numpy())

    if not ys_true:
        return pd.DataFrame()

    ytrue, ypred = np.vstack(ys_true), np.vstack(ys_pred)
    out = pd.DataFrame()
    for i, c in enumerate(display_target_cols):
        base = c.split("_ug")[0]
        out[f"{base}_true"] = ytrue[:, i]
        out[f"{base}_pred"] = ypred[:, i]
    return out


@torch.no_grad()
def compute_val_physical_mae(model, val_df, device):
    """
    Compute validation MAE in reconstructed physical space.

    This version is written for:
        cfg.target_mode == "residual_shift"

    Reconstruction:
        y_pred_total = y_ema_shift * (1 + pred_residual)
    """
    if getattr(cfg, "target_mode", "direct") != "residual_shift":
        raise ValueError(
            "compute_val_physical_mae currently supports only target_mode='residual_shift'."
        )

    model.eval()

    feats = resolved_feature_cols(cfg)
    targs = resolved_target_cols(cfg)
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)

    val_ds = WindowedDataset(val_df, feats, targs, window, stride)
    if len(val_ds) == 0:
        return float("inf")

    pin = torch.cuda.is_available()
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=pin,
    )

    scaler_path = Path(".runs/target_scaler.pkl")
    if not scaler_path.exists():
        raise FileNotFoundError(
            "Missing .runs/target_scaler.pkl. Run prepare_frames/train setup first."
        )

    with open(scaler_path, "rb") as f:
        target_scaler = pickle.load(f)

    preds_scaled = []
    for xb, yb in val_loader:
        xb = xb.to(device)
        yp = model(xb)
        if isinstance(yp, (tuple, list)):
            yp = yp[0]
        preds_scaled.append(yp.detach().cpu().numpy())

    if not preds_scaled:
        return float("inf")

    y_pred_scaled = np.vstack(preds_scaled)
    y_pred_resid = target_scaler.inverse_transform(y_pred_scaled)

    end_idxs = [e - 1 for (s, e) in val_ds.indices]

    # Current codebase uses single-target P_VA. Keep logic robust for multi-target too.
    maes = []
    for i, t0 in enumerate(cfg.target_cols):
        ema_col = f"{t0}_ema120_shift"
        true_col = f"{t0}_shift"

        if ema_col not in val_df.columns or true_col not in val_df.columns:
            raise KeyError(
                f"Validation reconstruction needs columns '{ema_col}' and '{true_col}'."
            )

        y_ema = val_df.iloc[end_idxs][ema_col].to_numpy(dtype=float)
        y_true = val_df.iloc[end_idxs][true_col].to_numpy(dtype=float)
        pred_resid = y_pred_resid[:, i]

        y_pred = y_ema * (1.0 + pred_resid)
        mae_i = float(np.mean(np.abs(y_true - y_pred)))
        maes.append(mae_i)

    return float(np.mean(maes))


def build_model(device, feats, targs):
    mt = cfg.model_type.lower()

    if mt in ("lstm", "gru"):
        return RNNRegressor(
            input_size=len(feats),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.model_type,
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=len(targs),
        ).to(device)

    if mt == "tcn":
        return TCNRegressor(
            input_size=len(feats),
            out_dim=len(targs),
            channels_list=getattr(cfg, "tcn_channels", (64, 64, 64)),
            kernel_size=getattr(cfg, "tcn_kernel_size", 3),
            dropout=cfg.dropout,
        ).to(device)

    raise ValueError(f"Unknown cfg.model_type={cfg.model_type}")


def train_one_split(train_df, val_df, device):
    os.makedirs("figures", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs(".runs", exist_ok=True)

    train_loader, val_loader = make_loaders(train_df, val_df, cfg)
    n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
    n_val = len(val_loader.dataset) if hasattr(val_loader, "dataset") else 0
    print(f"[debug] windows -> train: {n_train}, val: {n_val}")

    if n_train == 0 or n_val == 0:
        raise RuntimeError("No windows in train/val. Adjust window/shift/CSV length.")

    feats = resolved_feature_cols(cfg)
    targs = resolved_target_cols(cfg)
    model = build_model(device, feats, targs)

    run_name = f"{cfg.model_type}_hz{cfg.target_hz}_win{cfg.window_s}"
    writer: SummaryWriter = make_writer(logdir="runs", run_name=run_name)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    use_hetero = bool(getattr(cfg, "use_heteroscedastic", False)) and cfg.model_type.lower() == "tcn"
    if use_hetero:
        raise NotImplementedError(
            "This train.py version is set up for standard residual regression. "
            "Heteroscedastic mode is not enabled here."
        )
    else:
        criterion = HuberLoss(cfg.huber_delta) if cfg.use_huber else mae
        print("[info] using", "Huber loss" if cfg.use_huber else "MAE loss")

    # best_val now means best PHYSICAL validation MAE
    best_val = float("inf")
    best_state = None
    global_step = 0
    patience = getattr(cfg, "early_stopping_patience", 10)
    no_improve = 0

    epoch_train_losses, epoch_val_losses, grad_norms = [], [], []

    print(" starting training loop")
    try:
        for epoch in range(cfg.max_epochs):
            # ======= TRAIN =======
            model.train()
            batch_losses, batch_grad_norms = [], []

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)

                pred = model(xb)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

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

            # ======= VALIDATION LOSS IN RESIDUAL SPACE =======
            model.eval()
            vloss = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    vloss.append(criterion(pred, yb).item())

            mtrain = float(np.mean(batch_losses)) if batch_losses else float("nan")
            mval = float(np.mean(vloss)) if vloss else float("nan")
            epoch_train_losses.append(mtrain)
            epoch_val_losses.append(mval)
            grad_norms.append(float(np.mean(batch_grad_norms)) if batch_grad_norms else float("nan"))

            writer.add_scalar("loss/train", mtrain, epoch)
            writer.add_scalar("loss/val_residual", mval, epoch)

            # ======= VALIDATION MODEL SELECTION IN PHYSICAL SPACE =======
            val_phys_mae = compute_val_physical_mae(model, val_df, device)
            writer.add_scalar("metric/val_physical_mae", val_phys_mae, epoch)

            print(
                f" epoch {epoch:03d} | train {mtrain:.4f} | "
                f"val_loss {mval:.4f} | val_phys_mae {val_phys_mae:.4f}"
            )

            sch.step()

            if val_phys_mae < best_val:
                best_val = val_phys_mae
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[early stop] no val improvement for {patience} epochs; stopping at epoch {epoch}.")
                    break

    finally:
        plot_training_curves(epoch_train_losses, epoch_val_losses, "figures/training_curves.png")
        plot_grad_norms(grad_norms, "figures/grad_norms.png")
        writer.close()

    if best_state is None:
        raise RuntimeError("Training finished without a saved best_state.")

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
    print(f" best validation physical MAE = {best['val']:.6f}")

    # quick sanity check: reload best model and verify output dimensionality
    pin = torch.cuda.is_available()
    val_ds = WindowedDataset(splits["val"], feats, targs, window, stride)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=pin,
    )

    model = build_model(device, feats, targs)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    with torch.no_grad():
        xb0, _ = next(iter(val_loader))
        y0 = model(xb0.to(device))
        if isinstance(y0, (tuple, list)):
            y0 = y0[0]
        assert y0.shape[-1] == len(targs), f"Model outputs {y0.shape[-1]} dims, expected {len(targs)}"


if __name__ == "__main__":
    main()