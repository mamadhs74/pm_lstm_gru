import numpy as np
import pandas as pd

def regime_metrics(df_pred):
    out = {}
    for regime in ["idle","brake","cruise"]:
        if regime == "cruise":
            m = df_pred[(df_pred["idle"]==0)&(df_pred["brake_event"]==0)]
        else:
            m = df_pred[df_pred[regime]==1]
        if len(m) == 0:
            continue
        for t in ["PM10","PM2_5","PM1"]:
            e = np.abs(m[f"{t}_true"] - m[f"{t}_pred"]).mean()
            out[f"MAE_{regime}_{t}"] = e
    return out
