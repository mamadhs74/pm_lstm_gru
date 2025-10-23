# src/datasets_iter.py
import torch
from torch.utils.data import IterableDataset
import pandas as pd
from pathlib import Path
import numpy as np

class WindowedIterable(IterableDataset):
    """
    Streams Parquet shards and yields (X, y) windows without loading everything into RAM.
    Assumes df is preprocessed to target_hz and that transform() applies:
      add_regime_features -> scale -> build_residual_targets -> shift_targets -> warmup drop
    """
    def __init__(self, shard_paths, feature_cols, target_cols, window, stride, transform=None):
        super().__init__()
        self.shard_paths = list(map(Path, shard_paths))
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.window = window
        self.stride = stride
        self.transform = transform

    def __iter__(self):
        for spath in self.shard_paths:
            df = pd.read_parquet(spath)
            if self.transform is not None:
                df = self.transform(df)
            if df is None or len(df) == 0:
                continue

            Xv = df[self.feature_cols].values.astype(np.float32)
            Yv = df[self.target_cols].values.astype(np.float32)

            T = len(df)
            last = T - self.window
            i = 0
            while i <= last:
                x = torch.from_numpy(Xv[i:i+self.window])   # (T,F)
                y = torch.from_numpy(Yv[i+self.window-1])   # (3,) predict at window end
                yield x, y
                i += self.stride
