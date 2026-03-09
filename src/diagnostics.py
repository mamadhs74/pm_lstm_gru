import numpy as np
import pandas as pd

from src.config import cfg

def regime_metrics(m):
    targets = [c.split("_ug")[0] if "_ug" in c else c for c in cfg.target_cols]
    res = {}
    for t in targets:
        if f"{t}_true" not in m or f"{t}_pred" not in m:
            continue
        e = np.abs(m[f"{t}_true"] - m[f"{t}_pred"]).mean()
        res[t] = float(e)
    return res
