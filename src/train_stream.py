# src/train_stream.py
import os
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import cfg
from src.preprocess import (
    add_regime_features, build_residual_targets, shift_targets,
    downsample, leak_free_split, fit_transformers, apply_transformers
)
from src.datasets_iter import WindowedIterable
from src.datasets import WindowedDataset
from src.models import RNNRegressor
from src.metrics import mae, HuberLoss
from src.grad_viz import make_writer, log_gradients_tb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def make_stream_transform(cfg, scaler_path, scaler_type="standard"):
    if scaler_type == "standard":
        scaler = joblib.load(scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    def _transform(df):
        # if your raw shards are at 10 Hz, downsample here:
        df = downsample(df, cfg.orig_hz, cfg.target_hz)

        df = add_regime_features(df)
        # scale only the base features (engineered flags are added in Dataloader time)
        df[cfg.feature_cols] = scaler.transform(df[cfg.feature_cols])

        df = build_residual_targets(df, cfg)
        df = shift_targets(df, cfg)

        # warm-up drop to avoid EMA/shift boundary artifacts
        window = int(cfg.window_s * cfg.target_hz)
        shift  = int(cfg.target_shift_s * cfg.target_hz)
        warmup = window + shift + 5 * cfg.target_hz
        if len(df) > warmup:
            df = df.iloc[warmup:].reset_index(drop=True)
        else:
            df = df.iloc[0:0]  # empty
        return df
    return _transform

def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs(".runs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stream] device: {device}")

    shards = sorted(Path("data/parquet_shards").glob("*.parquet"))
    n = len(shards)
    purge = 1

    if n >= 6:
        train_end = int(0.70 * n)
        val_end   = int(0.85 * n)
        train_shards = shards[:train_end]
        val_shards   = shards[train_end + purge : val_end]
        test_shards  = shards[val_end + purge :]
    else:
        # Fallback for few shards: first n-2 train, one val, one test
        train_shards = shards[:-2] if n > 2 else shards[:1]
        val_shards   = shards[-2:-1] if n > 1 else []
        test_shards  = shards[-1:] if n > 0 else []
        purge = 0  # nothing to purge safely

    # choose scaler
    scaler_path = "data/processed/standard_scaler.joblib"  # or robust_scaler.joblib
    scaler_type = "standard"

    transform = make_stream_transform(cfg, scaler_path, scaler_type)
    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps
    feat_cols = cfg.feature_cols + ["idle", "brake_event", "dv", "jerk"]

    # ---- TRAIN: streaming iterable ----
    from src.datasets_iter import WindowedIterable
    train_iter = WindowedIterable(
        shard_paths=train_shards,
        feature_cols=feat_cols,
        target_cols=cfg.target_cols,
        window=window,
        stride=stride,
        transform=transform
    )
    train_loader = DataLoader(
        train_iter, batch_size=cfg.batch_size, num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # ---- VAL: read val shards into memory once, use your existing WindowedDataset
    def load_val_df():
        frames = []
        for sp in val_shards:
            df = pd.read_parquet(sp)
            df = downsample(df, cfg.orig_hz, cfg.target_hz)
            df = add_regime_features(df)
            frames.append(df)
        val_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        # fit scaler on TRAIN ONLY — we already fitted externally; here we just transform
        scaler = joblib.load(scaler_path)
        val_df[cfg.feature_cols] = scaler.transform(val_df[cfg.feature_cols])
        val_df = build_residual_targets(val_df, cfg)
        val_df = shift_targets(val_df, cfg)
        # align warmup like train
        warmup = window + int(cfg.target_shift_s * cfg.target_hz) + 5 * cfg.target_hz
        val_df = val_df.iloc[warmup:].reset_index(drop=True) if len(val_df) > warmup else val_df.iloc[0:0]
        return val_df

    val_df = load_val_df()
    from src.datasets import WindowedDataset
    val_ds = WindowedDataset(val_df, feat_cols, cfg.target_cols, window, stride)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=False, num_workers=2, pin_memory=torch.cuda.is_available())
    print(f"[stream] shards train/val/test = {len(train_shards)}/{len(val_shards)}/{len(test_shards)}")

    # ---- Model
    model = RNNRegressor(
        input_size=len(cfg.feature_cols)+4,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,
        conv_frontend=cfg.conv_frontend,
        conv_channels=cfg.conv_channels,
        out_dim=len(cfg.target_cols)
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    criterion = HuberLoss(cfg.huber_delta) if cfg.use_huber else mae

    writer = make_writer("runs", f"{cfg.model_type}_stream_hz{cfg.target_hz}_win{cfg.window_s}")
    best_val = float("inf")
    best_state = None
    global_step = 0

    print("[stream] training loop start")
    for epoch in range(cfg.max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            log_gradients_tb(writer, model, step=global_step)
            opt.step()
            train_losses.append(loss.item())
            global_step += 1
        sch.step()

        # VAL
        model.eval()
        vloss = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss.append(criterion(model(xb), yb).item())

        mtrain = float(np.mean(train_losses)) if train_losses else float("nan")
        mval   = float(np.mean(vloss))        if vloss        else float("nan")
        writer.add_scalar("loss/train", mtrain, epoch)
        writer.add_scalar("loss/val",   mval,   epoch)
        print(f"[epoch {epoch:03d}] train {mtrain:.4f}  val {mval:.4f}")

        if mval < best_val:
            best_val = mval
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt = f".runs/{cfg.model_type}_stream_best.pt"
    torch.save(best_state, ckpt)
    print(f"[stream] saved {ckpt}")
    writer.close()

if __name__ == "__main__":
    main()
