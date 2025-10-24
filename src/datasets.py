# src/datasets.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .config import resolved_feature_cols, resolved_target_cols, cfg

class WindowedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_cols, window_steps, stride, regime_weights=None):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_cols].values.astype(np.float32)

        self.idle  = df.get("idle", pd.Series(0, index=df.index)).values.astype(np.int64)
        self.brake = df.get("brake_event", pd.Series(0, index=df.index)).values.astype(np.int64)

        self.window = int(window_steps)
        self.stride = int(stride)

        self.indices = []
        self.weights = []
        last = len(df) - self.window
        i = 0
        while i <= last:
            s, e = i, i + self.window
            self.indices.append((s, e))
            w = 1.0
            if regime_weights is not None:
                if self.brake[e - 1] == 1:
                    w = regime_weights.get("brake", 1.0)
                elif self.idle[e - 1] == 1:
                    w = regime_weights.get("idle", 1.0)
                else:
                    w = regime_weights.get("cruise", 1.0)
            self.weights.append(w)
            i += self.stride

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s, e = self.indices[idx]
        x = torch.from_numpy(self.X[s:e])
        y = torch.from_numpy(self.y[e - 1])  # seq-to-one
        return x, y

    def sampler(self):
        return WeightedRandomSampler(self.weights, num_samples=len(self.weights), replacement=True)

def make_loaders(train_df, val_df, cfg):
    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)

    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    train_ds = WindowedDataset(train_df, feat_cols, targ_cols, window, stride, cfg.regime_weights)
    val_ds   = WindowedDataset(val_df,   feat_cols, targ_cols, window, stride, None)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_ds.sampler(),
                              drop_last=True, num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              drop_last=False, num_workers=2, pin_memory=pin)
    return train_loader, val_loader
