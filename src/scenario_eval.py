# scenario_eval.py
import pandas as pd
import torch
from pathlib import Path

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
from src.models import TCNRegressor, RNNRegressor
from src.visualize import plot_overlay


def _base_name(col: str) -> str:
    return col.split("_")[0]  # PM10_ug_per_m3 -> PM10


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


def run_scenario(excel_path, row_start, row_end,
                 ckpt_path, use_physics, scenario_name, tag):
    """
    excel_path : xlsx with raw Booster/Stopper data
    row_start, row_end : 0-based slice (see below)
    ckpt_path : path to trained model (with or without physics)
    use_physics : True / False (must match how the model was trained!)
    scenario_name : 'booster' or 'stopper'
    tag : 'phys' or 'plain' for filenames
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) set physics flag in config to match THIS checkpoint
    cfg.use_physics_baseline = use_physics

    # 2) read and cut the scenario
    raw = pd.read_excel(excel_path)
    raw = raw.iloc[row_start:row_end].reset_index(drop=True)

    # 3) full preprocessing (same as training/eval)
    splits = prepare_frames(raw)  # builds *_phys and *_resid_phys when use_physics=True

    # we use ONLY the test split (last 20 %) for prediction,
    # but since this file is just one scenario, test ~ scenario tail
    test_df = splits["test"].reset_index(drop=True)

    window = int(cfg.window_s * cfg.target_hz)
    stride = int(cfg.stride_steps)
    feat_cols = resolved_feature_cols(cfg)
    targ_cols = resolved_target_cols(cfg)

    ds = WindowedDataset(test_df, feat_cols, targ_cols, window, stride)
    if len(ds) == 0:
        print(f"[{scenario_name}/{tag}] No windows.")
        return

    # 4) build model and load checkpoint
    model = _build_model(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    display_names = cfg.target_cols  # ["PM10_ug_per_m3", "PM2_5_ug_per_m3", "PM1_ug_per_m3"]

    pred_df = _predict_df(model, ds, device, display_names, df_for_phys=test_df)

    # 5) overlay plot in PHYSICAL space if physics available
    #    (same logic as evaluate.py hybrid section)
    Path("figures").mkdir(exist_ok=True)

    base = "PM10"   # you can loop over all 3 if you want
    phys_col = f"{base}_phys"
    if phys_col in pred_df.columns:
        y_phys = pred_df[phys_col].to_numpy(float)
        y_true_tot = y_phys + pred_df[f"{base}_true"].to_numpy(float)
        y_pred_tot = y_phys + pred_df[f"{base}_pred"].to_numpy(float)

        df_hybrid = pd.DataFrame({
            f"{base}_true": y_true_tot,
            f"{base}_pred": y_pred_tot,
        })
        out_png = f"figures/{scenario_name}_{tag}_{base.lower()}_hybrid.png"
        plot_overlay(
            df_hybrid,
            out_png,
            title=f"{scenario_name.capitalize()} — {base} total ({tag}, hybrid vs truth)",
        )
        print("saved:", out_png)
    else:
        # case: physics OFF model – we’re already in total PM space
        df_overlay = pd.DataFrame({
            f"{base}_true": pred_df[f"{base}_true"],
            f"{base}_pred": pred_df[f"{base}_pred"],
        })
        out_png = f"figures/{scenario_name}_{tag}_{base.lower()}_plain.png"
        plot_overlay(
            df_overlay,
            out_png,
            title=f"{scenario_name.capitalize()} — {base} (plain vs truth)",
        )
        print("saved:", out_png)


if __name__ == "__main__":
    # Booster: lines 82:428  -> iloc[81:428]
    booster_xlsx = "data/raw/2025_09_03_10_06_44_Messfahrt_booster.xlsx"

    # Stopper: lines 143:489 -> iloc[142:489]
    stopper_xlsx = "data/raw/2025_09_03_10_14_44_Messfahrt_stoppper.xlsx"

    # --- with physics baseline ---
    run_scenario(
        excel_path=booster_xlsx,
        row_start=81, row_end=428,
        ckpt_path=".runs/tcn_phys_best.pt",
        use_physics=True,
        scenario_name="booster",
        tag="phys",
    )
    run_scenario(
        excel_path=stopper_xlsx,
        row_start=142, row_end=489,
        ckpt_path=".runs/tcn_phys_best.pt",
        use_physics=True,
        scenario_name="stopper",
        tag="phys",
    )

    # --- without physics baseline ---
    run_scenario(
        excel_path=booster_xlsx,
        row_start=81, row_end=428,
        ckpt_path=".runs/tcn_plain_best.pt",
        use_physics=False,
        scenario_name="booster",
        tag="plain",
    )
    run_scenario(
        excel_path=stopper_xlsx,
        row_start=142, row_end=489,
        ckpt_path=".runs/tcn_plain_best.pt",
        use_physics=False,
        scenario_name="stopper",
        tag="plain",
    )
