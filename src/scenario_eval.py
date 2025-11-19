# src/scenario_eval.py
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
    else:
        model = TCNRegressor(
            num_inputs=len(feats),
            num_channels=list(cfg.tcn_channels),
            kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
            out_dim=len(targs),
        ).to(device)
    return model


def _predict_df(model, ds, device, display_names, df_for_phys=None):
    from src.evaluate import _predict_df as base_predict_df
    return base_predict_df(model, ds, device, display_names, test_df_for_phys=df_for_phys)


def run_scenario(csv_path, row_start, row_end,
                 ckpt_path, use_physics, scenario_name, tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.use_physics_baseline = use_physics

    # 1) load cleaned Booster/Stopper
    raw = pd.read_csv(csv_path)
    raw = raw.iloc[row_start:row_end].reset_index(drop=True)

    # 2) for short scenarios, turn off purge to keep train split non-empty
    orig_purge = cfg.purge_seconds
    cfg.purge_seconds = 0
    splits = prepare_frames(raw)
    cfg.purge_seconds = orig_purge

    test_df = splits["test"].reset_index(drop=True)

    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    ds = WindowedDataset(test_df, feat_cols, targ_cols, window, stride)
    if len(ds) == 0:
        print(f"[{scenario_name}/{tag}] No windows.")
        return

    # 3) build model & load pretrained checkpoint from 1_clean
    model = _build_model(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    display_names = ["PM10_ug_per_m3"]
    pred_df = _predict_df(model, ds, device, display_names, df_for_phys=test_df)

    Path("figures").mkdir(exist_ok=True)
    base = "PM10"

    # 4) Build PM10 time-series for true vs predicted
    if use_physics and f"{base}_pred_total" in pred_df.columns:
        y_true = pred_df[f"{base}_true"].to_numpy(float)
        y_pred = pred_df[f"{base}_pred_total"].to_numpy(float)
        out_png = f"figures/{scenario_name}_{tag}_pm10_hybrid_overtime.png"
        title = f"{scenario_name.capitalize()} — PM10 total ({tag}, hybrid vs truth)"
    else:
        y_true = pred_df[f"{base}_true"].to_numpy(float)
        y_pred = pred_df[f"{base}_pred"].to_numpy(float)
        out_png = f"figures/{scenario_name}_{tag}_pm10_plain_overtime.png"
        title = f"{scenario_name.capitalize()} — PM10 (plain vs truth)"

    df_plot = pd.DataFrame({"PM10_true": y_true, "PM10_pred": y_pred})
    df_plot.index.name = "time_step"
    plot_overlay(df_plot, out_png, title=title)
    print("saved:", out_png)


if __name__ == "__main__":
    booster_csv = "data/processed/booster_clean.csv"
    stopper_csv = "data/processed/stopper_clean.csv"

    # MATLAB cuts: 'Booster' 82:428 → iloc[81:428], 'Stopper' 143:489 → iloc[142:489]
    booster_slice = (81, 428)
    stopper_slice = (142, 489)

    # --- with physics baseline model trained on 1_clean ---
    run_scenario(
        csv_path=booster_csv,
        row_start=booster_slice[0], row_end=booster_slice[1],
        ckpt_path=".runs/tcn_phys_best.pt",
        use_physics=True,
        scenario_name="booster",
        tag="phys",
    )
    run_scenario(
        csv_path=stopper_csv,
        row_start=stopper_slice[0], row_end=stopper_slice[1],
        ckpt_path=".runs/tcn_phys_best.pt",
        use_physics=True,
        scenario_name="stopper",
        tag="phys",
    )

    # --- without physics baseline ---
    run_scenario(
        csv_path=booster_csv,
        row_start=booster_slice[0], row_end=booster_slice[1],
        ckpt_path=".runs/tcn_plain_best.pt",
        use_physics=False,
        scenario_name="booster",
        tag="plain",
    )
    run_scenario(
        csv_path=stopper_csv,
        row_start=stopper_slice[0], row_end=stopper_slice[1],
        ckpt_path=".runs/tcn_plain_best.pt",
        use_physics=False,
        scenario_name="stopper",
        tag="plain",
    )
