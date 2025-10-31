# src/report_more.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def _ensure_dir(p): Path(p).parent.mkdir(parents=True, exist_ok=True)

def plot_hybrid_vs_physics(df, out_png, target="PM10"):
    """
    df needs columns:
      - PM10_true, PM10_pred_resid (network residual prediction), PM10_phys (physics baseline)
    If your pipeline outputs only residual preds, compute hybrid = phys + pred_resid.
    """
    base = target
    true = df[f"{base}_true"].values
    pred_resid = df.get(f"{base}_pred_resid", None)
    phys = df.get(f"{base}_phys", None)
    if phys is None:
        raise ValueError("Expected physics baseline column f'{base}_phys' not found.")
    if pred_resid is None:
        # If your CSV has hybrid already as PM10_pred, derive pred_resid = PM10_pred - phys
        if f"{base}_pred" in df.columns:
            pred_resid = df[f"{base}_pred"].values - phys
        else:
            raise ValueError("Need either PM10_pred_resid or PM10_pred + PM10_phys.")

    hybrid = phys + pred_resid
    x = np.arange(len(true))

    _ensure_dir(out_png)
    plt.figure(figsize=(12,4))
    plt.plot(x, true, label="truth", linewidth=1.2)
    plt.plot(x, phys, label="physics", linestyle="--")
    plt.plot(x, hybrid, label="hybrid", alpha=0.9)
    plt.title(f"{base} — physics vs hybrid vs truth")
    plt.xlabel("time (samples @ target_hz)")
    plt.ylabel("µg/m³")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def binned_mae(df, col, bins, targets, out_png, label=None):
    """
    MAE(residual) per bin of `col`. Needs {PM*_true, PM*_pred or _pred_resid, PM*_phys}.
    Uses hybrid residual = truth - hybrid.
    """
    _ensure_dir(out_png)
    d = df.copy()
    d["bin"] = pd.cut(d[col], bins=bins, include_lowest=True)
    xs, maes = [], {t:[] for t in targets}
    # compute residual from hybrid
    for t in targets:
        base = t.split("_ug")[0]
        if f"{base}_phys" not in d.columns:
            raise ValueError(f"Need {base}_phys in df for binned_mae.")
        phys = d[f"{base}_phys"].values
        if f"{base}_pred_resid" in d.columns:
            hybrid = phys + d[f"{base}_pred_resid"].values
        elif f"{base}_pred" in d.columns:
            hybrid = d[f"{base}_pred"].values
        else:
            raise ValueError(f"Need {base}_pred_resid or {base}_pred.")
        resid = d[f"{base}_true"].values - hybrid
        d[f"{base}_resid_hybrid"] = resid

    grp = d.groupby("bin")
    for b, g in grp:
        xs.append(str(b))
        for t in targets:
            base = t.split("_ug")[0]
            maes[t].append(np.mean(np.abs(g[f"{base}_resid_hybrid"].values)))

    plt.figure(figsize=(12,4))
    for t in targets:
        plt.plot(xs, maes[t], marker="o", label=t.replace("_ug_per_m3",""))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAE (hybrid residual)")
    ttl = f"MAE vs {col}" + (f" — {label}" if label else "")
    plt.title(ttl)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def brake_onset_average(df, brake_flag="brake_event", halfwin=50, targets=("PM10_ug_per_m3",)):
    """
    Find indices where brake_event rises 0->1; cut windows [i-halfwin, i+halfwin]; 
    compute mean truth/phys/hybrid across events.
    Returns {target: (tvec, mean_truth, mean_phys, mean_hybrid)}
    """
    out = {}
    b = df[brake_flag].values.astype(int)
    onset = np.where((b[1:]==1) & (b[:-1]==0))[0] + 1
    for t in targets:
        base = t.split("_ug")[0]
        if f"{base}_phys" not in df.columns: continue
        phys = df[f"{base}_phys"].values
        if f"{base}_pred_resid" in df.columns:
            hybrid = phys + df[f"{base}_pred_resid"].values
        elif f"{base}_pred" in df.columns:
            hybrid = df[f"{base}_pred"].values
        else:
            continue
        truth = df[f"{base}_true"].values
        stks = []
        for i in onset:
            s, e = i-halfwin, i+halfwin+1
            if s<0 or e>len(df): continue
            stks.append(np.stack([truth[s:e], phys[s:e], hybrid[s:e]], axis=0))
        if not stks: 
            continue
        arr = np.stack(stks, axis=0)   # (events, 3, T)
        m = arr.mean(axis=0)           # (3, T)
        tvec = np.arange(-halfwin, halfwin+1)
        out[t] = (tvec, m[0], m[1], m[2])
    return out

def plot_brake_onset_average(avg_dict, out_png, target="PM10_ug_per_m3"):
    _ensure_dir(out_png)
    tvec, truth, phys, hybrid = avg_dict[target]
    plt.figure(figsize=(8,4))
    plt.plot(tvec, truth, label="truth")
    plt.plot(tvec, phys,  label="physics", linestyle="--")
    plt.plot(tvec, hybrid,label="hybrid")
    plt.axvline(0, color="k", linewidth=0.8)
    plt.title(f"Brake onset average — {target.replace('_ug_per_m3','')}")
    plt.xlabel("samples around brake onset (t=0)")
    plt.ylabel("µg/m³")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def residual_autocorr(df, target, out_png, max_lag=150):
    base = target.split("_ug")[0]
    if f"{base}_phys" not in df.columns: 
        raise ValueError("Need physics baseline.")
    if f"{base}_pred_resid" in df.columns:
        resid = df[f"{base}_true"] - (df[f"{base}_phys"] + df[f"{base}_pred_resid"])
    elif f"{base}_pred" in df.columns:
        resid = df[f"{base}_true"] - df[f"{base}_pred"]
    else:
        raise ValueError("Need prediction columns.")

    r = np.correlate(resid - resid.mean(), resid - resid.mean(), mode="full")
    acf = r[r.size//2: r.size//2+max_lag]
    acf = acf / acf[0]

    _ensure_dir(out_png)
    plt.figure(figsize=(7,3))
    plt.stem(np.arange(len(acf)), acf, use_line_collection=True)
    plt.title(f"Residual ACF — {target.replace('_ug_per_m3','')}")
    plt.xlabel("lag (samples)")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def rolling_mae(df, target, out_png, win=60):
    base = target.split("_ug")[0]
    if f"{base}_phys" not in df.columns: return
    if f"{base}_pred_resid" in df.columns:
        hybrid = df[f"{base}_phys"] + df[f"{base}_pred_resid"]
    elif f"{base}_pred" in df.columns:
        hybrid = df[f"{base}_pred"]
    else:
        return
    resid = (df[f"{base}_true"] - hybrid).abs().rolling(win, min_periods=1).mean()
    _ensure_dir(out_png)
    plt.figure(figsize=(12,3.5))
    plt.plot(resid.values)
    plt.title(f"Rolling MAE (window={win}) — {target.replace('_ug_per_m3','')}")
    plt.xlabel("time (samples)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def permutation_feature_importance(val_df, feats, targets, model, loader, device, out_png):
    """
    Simple permutation importance on val: permute one feature column in-memory, re-run loader once per feat.
    Expect loader to be rebuilt per permutation; we implement a light version here by modifying a copy of val_df.
    """
    import torch
    from src.datasets import WindowedDataset
    from torch.utils.data import DataLoader

    window = loader.dataset.window
    stride = loader.dataset.stride
    pin = torch.cuda.is_available()

    # baseline MAE
    def eval_df(df):
        ds = WindowedDataset(df, feats, targets, window, stride)
        dl = DataLoader(ds, batch_size=loader.batch_size, shuffle=False, drop_last=False, num_workers=2, pin_memory=pin)
        maes = []
        model.eval()
        with torch.no_grad():
            ys, yp = [], []
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                ys.append(yb.cpu().numpy()); yp.append(yhat.cpu().numpy())
        import numpy as np
        ys = np.vstack(ys); yp = np.vstack(yp)
        return np.mean(np.abs(ys-yp))

    base_mae = eval_df(val_df)
    drops = []
    rng = np.random.default_rng(42)
    for f in feats:
        dfp = val_df.copy()
        dfp[f] = rng.permutation(dfp[f].values)  # permute column
        drops.append(eval_df(dfp) - base_mae)

    _ensure_dir(out_png)
    order = np.argsort(drops)[::-1]
    plt.figure(figsize=(8,5))
    plt.bar([feats[i] for i in order], [drops[i] for i in order])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("ΔMAE (permute)")
    plt.title("Permutation Feature Importance (validation)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
