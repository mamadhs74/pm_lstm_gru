# scripts/predict_booster_stopper.py
import pandas as pd
import torch
from pathlib import Path

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
from src.models import TCNRegressor, RNNRegressor
from src.visualize import plot_overlay


def _build_model(device):
    feats = resolved_feature_cols(cfg)
    targs = resolved_target_cols(cfg)
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
    else:
        return TCNRegressor(
            num_inputs=len(feats),
            num_channels=list(cfg.tcn_channels),
            kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            out_dim=len(targs),
        ).to(device)


def predict_one(clean_csv, ckpt_path, tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Predicting {clean_csv} with model {ckpt_path} ===")

    # IMPORTANT for short scenario traces:
    # do not use a big purge gap, otherwise TRAIN becomes empty
    cfg.purge_seconds = 0.0

    raw = pd.read_csv(clean_csv)

    # FULL PIPELINE (now without purge gap)
    splits = prepare_frames(raw)

    test_df = splits["test"].reset_index(drop=True)

    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)

    ds = WindowedDataset(test_df, feat_cols, targ_cols, window, stride)
    if len(ds) == 0:
        print("[WARN] No windows!")
        return

    # LOAD MODEL
    model = _build_model(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # PREDICT
    from src.evaluate import _predict_df
    pred_df = _predict_df(model, ds, device, cfg.target_cols, test_df_for_phys=test_df)

    # ALWAYS PLOT PM10
    true_pm = pred_df["PM10_true"]
    pred_pm = pred_df["PM10_pred"]

    df_plot = pd.DataFrame({
        "PM10_true": true_pm,
        "PM10_pred": pred_pm,
    })

    out = f"figures/{Path(clean_csv).stem}_{tag}_pm10_overlay.png"
    plot_overlay(df_plot, out, title=f"{Path(clean_csv).stem} — PM10 ({tag})")

    print("Saved:", out)


if __name__ == "__main__":
    # clean CSVs from your Option B script
    booster = "data/processed/booster_clean.csv"
    stopper = "data/processed/stopper_clean.csv"

    # paths to trained models
    plain = ".runs/tcn_best.pt"
    phys  = ".runs/tcn_best.pt"   # or comment out phys calls for now

    # PREDICT
    predict_one(booster, plain, "plain")
    predict_one(stopper, plain, "plain")

    predict_one(booster, phys,  "phys")
    predict_one(stopper, phys,  "phys")
