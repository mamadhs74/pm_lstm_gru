import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class WindowedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_cols, window_steps, stride, regime_weights=None):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[[c + "_resid_shift" for c in target_cols]].values.astype(np.float32)
        self.idle = df.get("idle", pd.Series(0, index=df.index)).values.astype(np.int64)
        self.brake = df.get("brake_event", pd.Series(0, index=df.index)).values.astype(np.int64)
        self.window = window_steps
        self.stride = stride
        self.indices = []
        self.weights = []
        for start in range(0, len(df) - window_steps, stride):
            end = start + window_steps
            self.indices.append((start, end))
            # regime weight by last sample regime
            w = 1.0
            if regime_weights is not None:
                if self.brake[end - 1] == 1:
                    w = regime_weights.get("brake", 1.0)
                elif self.idle[end - 1] == 1:
                    w = regime_weights.get("idle", 1.0)
                else:
                    w = regime_weights.get("cruise", 1.0)
            self.weights.append(w)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s, e = self.indices[idx]
        x = self.X[s:e]
        y = self.y[e - 1]  # sequence-to-one
        return torch.from_numpy(x), torch.from_numpy(y)

    def sampler(self):
        return WeightedRandomSampler(self.weights, num_samples=len(self.weights), replacement=True)


def make_loaders(train_df, val_df, cfg):
    window = int(cfg.window_s * cfg.target_hz)
    stride = cfg.stride_steps
    feat_cols = cfg.feature_cols + ["idle", "brake_event", "dv", "jerk"]
    train_ds = WindowedDataset(train_df, feat_cols, cfg.target_cols, window, stride, cfg.regime_weights)
    val_ds = WindowedDataset(val_df, feat_cols, cfg.target_cols, window, stride, None)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_ds.sampler(),
        drop_last=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
