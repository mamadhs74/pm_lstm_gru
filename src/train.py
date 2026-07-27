import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import GaussianNLLLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from rich import print

from .grad_viz import make_writer, log_gradients_tb
from .config import (
    cfg,
    resolved_feature_cols,
    resolved_target_cols,
)
from .preprocess import prepare_frames
from .datasets import make_loaders, WindowedDataset
from .models import RNNRegressor, TCNRegressor
from .metrics import mae, HuberLoss
from .visualize import (
    plot_training_curves,
    plot_grad_norms,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.dropna(subset=cfg.target_cols)


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
        f"Unknown cfg.model_type={cfg.model_type}"
    )


def train_one_split(
    train_df,
    val_df,
    device,
):
    run_dir = (
        Path("publication_results")
        / "training"
        / cfg.target_mode
        / f"seed_{cfg.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(
        train_df,
        val_df,
        cfg,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    print(
        f"[debug] windows -> train: {n_train}, "
        f"val: {n_val}"
    )

    if n_train == 0 or n_val == 0:
        raise RuntimeError(
            "No train/validation windows."
        )

    model = build_model(device)

    run_name = (
        f"{cfg.model_type}_"
        f"{cfg.target_mode}_"
        f"seed{cfg.seed}_"
        f"hz{cfg.target_hz}_"
        f"win{cfg.window_s}"
    )

    writer: SummaryWriter = make_writer(
        logdir="runs",
        run_name=run_name,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_epochs,
    )

    use_heterogeneous = (
        bool(cfg.use_heteroscedastic)
        and cfg.model_type.lower() == "tcn"
    )

    if use_heterogeneous:
        criterion = GaussianNLLLoss(full=True)
        print("[info] using Gaussian NLL")
    else:
        criterion = (
            HuberLoss(cfg.huber_delta)
            if cfg.use_huber
            else mae
        )
        print(
            "[info] using",
            "Huber loss" if cfg.use_huber else "MAE",
        )

    best_val = float("inf")
    best_state = None
    best_epoch = None
    no_improve = 0
    global_step = 0

    train_losses = []
    val_losses = []
    gradient_norms = []

    try:
        for epoch in range(cfg.max_epochs):
            model.train()
            batch_losses = []
            batch_gradient_norms = []

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)

                if use_heterogeneous:
                    mu, log_variance = model(xb)
                    variance = (
                        torch.exp(log_variance)
                        .clamp_min(1e-6)
                    )
                    loss = criterion(
                        mu,
                        yb,
                        variance,
                    )
                else:
                    prediction = model(xb)
                    loss = criterion(
                        prediction,
                        yb,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
                optimizer.step()

                if global_step % 50 == 0:
                    log_gradients_tb(
                        writer,
                        model,
                        step=global_step,
                    )

                squared_norm = 0.0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        gradient = (
                            parameter.grad
                            .detach()
                            .flatten()
                        )
                        squared_norm += float(
                            torch.dot(
                                gradient,
                                gradient,
                            )
                        )

                batch_gradient_norms.append(
                    np.sqrt(squared_norm)
                )
                batch_losses.append(
                    float(loss.detach().cpu())
                )
                global_step += 1

            model.eval()
            validation_batch_losses = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    if use_heterogeneous:
                        mu, log_variance = model(xb)
                        variance = (
                            torch.exp(log_variance)
                            .clamp_min(1e-6)
                        )
                        validation_loss = criterion(
                            mu,
                            yb,
                            variance,
                        )
                    else:
                        validation_loss = criterion(
                            model(xb),
                            yb,
                        )

                    validation_batch_losses.append(
                        float(validation_loss.item())
                    )

            mean_train = float(
                np.mean(batch_losses)
            )
            mean_val = float(
                np.mean(validation_batch_losses)
            )
            mean_gradient = float(
                np.mean(batch_gradient_norms)
            )

            train_losses.append(mean_train)
            val_losses.append(mean_val)
            gradient_norms.append(mean_gradient)

            writer.add_scalar(
                "loss/train",
                mean_train,
                epoch,
            )
            writer.add_scalar(
                "loss/val",
                mean_val,
                epoch,
            )

            print(
                f"epoch {epoch:03d} | "
                f"train {mean_train:.4f} | "
                f"val {mean_val:.4f}"
            )

            scheduler.step()

            if mean_val < best_val:
                best_val = mean_val
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu()
                    for key, value
                    in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

                if (
                    no_improve
                    >= cfg.early_stopping_patience
                ):
                    print(
                        "[early stop] no validation "
                        f"improvement for "
                        f"{cfg.early_stopping_patience} "
                        f"epochs; stopping at {epoch}."
                    )
                    break

    finally:
        history = pd.DataFrame({
            "epoch": range(len(train_losses)),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "gradient_norm": gradient_norms,
        })
        history.to_csv(
            run_dir / "training_history.csv",
            index=False,
        )

        plot_training_curves(
            train_losses,
            val_losses,
            str(run_dir / "training_curves.png"),
        )
        plot_grad_norms(
            gradient_norms,
            str(run_dir / "gradient_norms.png"),
        )
        writer.close()

    if best_state is None:
        raise RuntimeError(
            "Training produced no valid checkpoint."
        )

    return {
        "val": best_val,
        "state": best_state,
        "best_epoch": best_epoch,
        "run_dir": run_dir,
    }


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

    maximum_epochs_override = os.getenv(
        "RUN_MAX_EPOCHS"
    )
    if maximum_epochs_override:
        cfg.max_epochs = int(
            maximum_epochs_override
        )

    set_seed(cfg.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)
    print("seed:", cfg.seed)
    print("target mode:", cfg.target_mode)

    frame = load_csv(cfg.data_path)
    print(
        f"loaded {len(frame)} rows "
        f"from {cfg.data_path}"
    )

    splits = prepare_frames(frame)

    window = int(
        cfg.window_s * cfg.target_hz
    )
    stride = int(cfg.stride_steps)
    features = resolved_feature_cols(cfg)
    targets = resolved_target_cols(cfg)

    counts = {}
    for name in [
        "train",
        "val",
        "test",
        "stationary_test",
    ]:
        dataset = WindowedDataset(
            splits[name],
            features,
            targets,
            window,
            stride,
        )
        counts[name] = {
            "rows": int(len(splits[name])),
            "windows": int(len(dataset)),
        }

    print("rows/windows:", counts)

    result = train_one_split(
        splits["train"],
        splits["val"],
        device,
    )

    checkpoint_name = (
        f"{cfg.model_type}_"
        f"{cfg.target_mode}_"
        f"seed{cfg.seed}_best.pt"
    )
    checkpoint_path = (
        Path(".runs") / checkpoint_name
    )

    torch.save(
        result["state"],
        checkpoint_path,
    )
    torch.save(
        result["state"],
        result["run_dir"] / "checkpoint.pt",
    )

    manifest = {
        "seed": cfg.seed,
        "target_mode": cfg.target_mode,
        "model_type": cfg.model_type,
        "feature_cols": list(features),
        "resolved_target_cols": list(targets),
        "orig_hz": cfg.orig_hz,
        "target_hz": cfg.target_hz,
        "window_s": cfg.window_s,
        "window_samples": window,
        "target_shift_s": cfg.target_shift_s,
        "forecast_samples": int(
            cfg.target_shift_s
            * cfg.target_hz
        ),
        "stride_steps": cfg.stride_steps,
        "tcn_channels": list(
            cfg.tcn_channels
        ),
        "tcn_kernel_size": (
            cfg.tcn_kernel_size
        ),
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "maximum_epochs": cfg.max_epochs,
        "early_stopping_patience": (
            cfg.early_stopping_patience
        ),
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "loss": (
            "Huber"
            if cfg.use_huber
            else "MAE"
        ),
        "huber_delta": cfg.huber_delta,
        "best_epoch": result["best_epoch"],
        "best_validation_loss": result["val"],
        "checkpoint": str(checkpoint_path),
        "split_counts": counts,
        "split_manifest": splits.get(
            "_split_manifest",
            {},
        ),
        "device": str(device),
    }

    with open(
        result["run_dir"] / "run_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )

    # Check that the saved state reloads correctly.
    validation_dataset = WindowedDataset(
        splits["val"],
        features,
        targets,
        window,
        stride,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    check_model = build_model(device)
    check_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    with torch.no_grad():
        xb, _ = next(iter(validation_loader))
        output = check_model(xb.to(device))

    assert output.shape[-1] == len(targets)

    print(
        "saved best model to",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
