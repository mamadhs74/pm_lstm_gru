import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader

from src.config import (
    cfg,
    resolved_feature_cols,
    resolved_target_cols,
)
from src.preprocess import prepare_frames
from src.datasets import WindowedDataset
from src.models import (
    RNNRegressor,
    TCNRegressor,
)


def build_model(device):
    features = resolved_feature_cols(cfg)
    targets = resolved_target_cols(cfg)

    if cfg.model_type.lower() in ("lstm", "gru"):
        return RNNRegressor(
            input_size=len(features),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.model_type,
            conv_frontend=cfg.conv_frontend,
            conv_channels=cfg.conv_channels,
            out_dim=len(targets),
        ).to(device)

    if cfg.model_type.lower() == "tcn":
        return TCNRegressor(
            input_size=len(features),
            out_dim=len(targets),
            channels_list=cfg.tcn_channels,
            kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
        ).to(device)

    raise ValueError(
        f"Unknown model type: {cfg.model_type}"
    )


def metric_row(
    model_name,
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mae_value = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )
    rmse_value = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )
    r2_value = float(
        r2_score(
            y_true,
            y_pred,
        )
    )
    value_range = float(
        np.max(y_true) - np.min(y_true)
    )
    nrmse_value = (
        float(rmse_value / value_range)
        if value_range > 0
        else float("nan")
    )
    bias_value = float(
        np.mean(y_pred - y_true)
    )

    return {
        "model": model_name,
        "mae": mae_value,
        "rmse": rmse_value,
        "r2": r2_value,
        "nrmse_range": nrmse_value,
        "bias": bias_value,
        "n_predictions": int(len(y_true)),
    }


@torch.no_grad()
def predict_scaled(
    model,
    dataset,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    truths = []
    predictions = []

    model.eval()

    for xb, yb in loader:
        output = model(xb.to(device))

        if isinstance(output, (tuple, list)):
            output = output[0]

        truths.append(
            yb.detach().cpu().numpy()
        )
        predictions.append(
            output.detach().cpu().numpy()
        )

    if not truths:
        raise RuntimeError(
            "Evaluation dataset contains no windows."
        )

    return (
        np.vstack(truths),
        np.vstack(predictions),
    )


def main():
    cfg.seed = int(
        os.getenv(
            "RUN_SEED",
            str(cfg.seed),
        )
    )
    cfg.target_mode = os.getenv(
        "RUN_TARGET_MODE",
        cfg.target_mode,
    )
    cfg.evaluation_scope = os.getenv(
        "RUN_EVAL_SCOPE",
        cfg.evaluation_scope,
    )

    if cfg.evaluation_scope not in {
        "active",
        "stationary",
    }:
        raise ValueError(
            "RUN_EVAL_SCOPE must be active "
            "or stationary."
        )

    split_key = (
        "test"
        if cfg.evaluation_scope == "active"
        else "stationary_test"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    raw = (
        pd.read_csv(cfg.data_path)
        .dropna(subset=cfg.target_cols)
    )
    splits = prepare_frames(raw)

    evaluation_frame = (
        splits[split_key]
        .reset_index(drop=True)
    )

    features = resolved_feature_cols(cfg)
    targets = resolved_target_cols(cfg)
    window = int(
        cfg.window_s * cfg.target_hz
    )
    stride = int(cfg.stride_steps)

    dataset = WindowedDataset(
        evaluation_frame,
        features,
        targets,
        window,
        stride,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"{split_key} contains no windows."
        )

    model = build_model(device)

    checkpoint_path = Path(".runs") / (
        f"{cfg.model_type}_"
        f"{cfg.target_mode}_"
        f"seed{cfg.seed}_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    y_true_scaled, y_pred_scaled = (
        predict_scaled(
            model,
            dataset,
            device,
        )
    )

    target_scaler = splits["_target_scaler"]

    y_true_model = (
        target_scaler
        .inverse_transform(y_true_scaled)
        .ravel()
    )
    y_pred_model = (
        target_scaler
        .inverse_transform(y_pred_scaled)
        .ravel()
    )

    end_indices = np.asarray(
        [
            end - 1
            for start, end in dataset.indices
        ],
        dtype=int,
    )

    base_target = cfg.target_cols[0]

    physical_truth_column = (
        f"{base_target}_future_physical"
    )
    ema_column = (
        f"{base_target}_ema120_shift"
    )

    required_columns = [
        base_target,
        physical_truth_column,
        ema_column,
    ]
    missing = [
        column
        for column in required_columns
        if column not in evaluation_frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing evaluation columns: {missing}"
        )

    y_true = (
        evaluation_frame
        .iloc[end_indices][physical_truth_column]
        .to_numpy(dtype=float)
    )

    # Persistence forecast: current measurement at t.
    y_persistence = (
        evaluation_frame
        .iloc[end_indices][base_target]
        .to_numpy(dtype=float)
    )

    # Causal EMA available at prediction time t.
    y_ema = (
        evaluation_frame
        .iloc[end_indices][ema_column]
        .to_numpy(dtype=float)
    )

    if cfg.target_mode == "direct":
        y_model = y_pred_model
        model_name = "direct_tcn"

        direct_truth_difference = float(
            np.max(
                np.abs(
                    y_true_model - y_true
                )
            )
        )

        if direct_truth_difference > 1e-5:
            raise RuntimeError(
                "Direct-target inverse transform "
                "does not match physical truth."
            )

    elif cfg.target_mode == "residual_shift":
        y_model = (
            y_ema
            + (np.abs(y_ema) + 1e-6)
            * y_pred_model
        )
        model_name = "ema_residual_tcn"

    else:
        raise ValueError(
            "Publication evaluation supports "
            "direct and residual_shift."
        )

    output_dir = (
        Path("publication_results")
        / cfg.evaluation_scope
        / cfg.target_mode
        / f"seed_{cfg.seed}"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.DataFrame({
        "original_end_row_in_split": end_indices,
        "y_true": y_true,
        "persistence": y_persistence,
        "causal_ema": y_ema,
        model_name: y_model,
    })

    for column in [
        "Time_since_midnight_UTC",
        "Zeitstempel",
        "X_Achse_[s]",
        "time_s",
        "speed_kmh_physical",
        "acc_long_ms2_physical",
        "acc_lat_ms2_physical",
        "steer_rate_deg_s_physical",
    ]:
        if column in evaluation_frame.columns:
            predictions[column] = (
                evaluation_frame
                .iloc[end_indices][column]
                .to_numpy()
            )

    predictions.to_csv(
        output_dir
        / "test_predictions_physical.csv",
        index=False,
    )

    rows = [
        metric_row(
            "persistence",
            y_true,
            y_persistence,
        ),
        metric_row(
            "causal_ema",
            y_true,
            y_ema,
        ),
        metric_row(
            model_name,
            y_true,
            y_model,
        ),
    ]

    metrics = pd.DataFrame(rows)
    metrics.insert(
        0,
        "target",
        base_target,
    )
    metrics.insert(
        1,
        "evaluation_scope",
        cfg.evaluation_scope,
    )
    metrics.insert(
        2,
        "target_mode",
        cfg.target_mode,
    )
    metrics.insert(
        3,
        "seed",
        cfg.seed,
    )

    metrics.to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    speed = (
        predictions["speed_kmh_physical"]
        if "speed_kmh_physical"
        in predictions.columns
        else pd.Series(dtype=float)
    )
    acc_long = (
        predictions["acc_long_ms2_physical"]
        if "acc_long_ms2_physical"
        in predictions.columns
        else pd.Series(dtype=float)
    )
    acc_lat = (
        predictions["acc_lat_ms2_physical"]
        if "acc_lat_ms2_physical"
        in predictions.columns
        else pd.Series(dtype=float)
    )

    regime_summary = {
        "evaluation_scope": cfg.evaluation_scope,
        "split_key": split_key,
        "rows": int(len(evaluation_frame)),
        "windows": int(len(dataset)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true)),
        "target_min": float(np.min(y_true)),
        "target_max": float(np.max(y_true)),
        "target_above_100": int(
            np.sum(y_true > 100)
        ),
        "target_above_500": int(
            np.sum(y_true > 500)
        ),
    }

    if len(speed):
        regime_summary.update({
            "speed_mean_kmh": float(
                speed.mean()
            ),
            "speed_max_kmh": float(
                speed.max()
            ),
            "moving_predictions_speed_ge_1": int(
                (speed >= 1.0).sum()
            ),
        })

    if len(acc_long):
        regime_summary[
            "acc_long_abs_max_ms2"
        ] = float(
            acc_long.abs().max()
        )

    if len(acc_lat):
        regime_summary[
            "acc_lat_abs_max_ms2"
        ] = float(
            acc_lat.abs().max()
        )

    with open(
        output_dir / "regime_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            regime_summary,
            handle,
            indent=2,
        )

    run_manifest = {
        "checkpoint": str(checkpoint_path),
        "seed": cfg.seed,
        "target_mode": cfg.target_mode,
        "evaluation_scope": (
            cfg.evaluation_scope
        ),
        "split_key": split_key,
        "feature_cols": list(features),
        "resolved_target_cols": list(targets),
        "window_s": cfg.window_s,
        "window_samples": window,
        "forecast_horizon_s": (
            cfg.target_shift_s
        ),
        "forecast_samples": int(
            cfg.target_shift_s
            * cfg.target_hz
        ),
        "stride_steps": cfg.stride_steps,
        "batch_size": cfg.batch_size,
        "split_manifest": splits.get(
            "_split_manifest",
            {},
        ),
    }

    with open(
        output_dir / "run_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_manifest,
            handle,
            indent=2,
        )

    sample_count = min(
        500,
        len(y_true),
    )

    plt.figure(figsize=(11, 4))
    plt.plot(
        y_true[:sample_count],
        label="Measured",
    )
    plt.plot(
        y_persistence[:sample_count],
        label="Persistence",
    )
    plt.plot(
        y_ema[:sample_count],
        label="Causal EMA",
    )
    plt.plot(
        y_model[:sample_count],
        label=model_name,
    )
    plt.xlabel("Prediction")
    plt.ylabel(base_target)
    plt.title(
        f"{base_target} — "
        f"{cfg.evaluation_scope} — "
        f"seed {cfg.seed}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "timeseries.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_true,
        y_model,
        s=12,
        alpha=0.5,
    )
    minimum = float(
        min(y_true.min(), y_model.min())
    )
    maximum = float(
        max(y_true.max(), y_model.max())
    )
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--",
    )
    plt.xlabel("Measured")
    plt.ylabel("Predicted")
    plt.title(model_name)
    plt.tight_layout()
    plt.savefig(
        output_dir / "parity.png",
        dpi=200,
    )
    plt.close()

    print("\n=== EVALUATION RESULTS ===")
    print(metrics.to_string(index=False))
    print("\n=== REGIME SUMMARY ===")
    print(
        json.dumps(
            regime_summary,
            indent=2,
        )
    )
    print("\nWrote:", output_dir)


if __name__ == "__main__":
    main()
