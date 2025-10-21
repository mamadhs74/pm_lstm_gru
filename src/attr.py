# src/attr.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

def integrated_grads_feature_importance(
    model,
    sample_batch: torch.Tensor,          # (B, T, F)
    feature_names,
    target_index: int = 0,                # 0: PM10, 1: PM2_5, 2: PM1
    steps: int = 32,
    device: str = "cpu",
    use_absolute: bool = True,            # magnitude vs signed
    internal_batch_size: int | None = None,
    return_delta: bool = False,
    baseline: torch.Tensor | None = None  # optional custom baseline
):
    """
    Integrated Gradients per input feature over the time window.

    Returns:
      series: pd.Series length F (higher = more important)
      (optional) delta: Captum convergence delta (lower = better attribution completeness)
    """
    model.eval()
    x = sample_batch.to(device)
    assert x.ndim == 3, f"Expected (B,T,F), got {tuple(x.shape)}"
    B, T, F = x.shape

    # feature name checks
    assert len(feature_names) == F, f"feature_names length {len(feature_names)} != F {F}"

    # inputs must require grad for IG
    x = x.requires_grad_()

    # default baseline: zeros (common Captum choice)
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device)
        assert baseline.shape == x.shape, "baseline must match input shape"

    def forward_func(inp):
        y = model(inp)            # (B, 3)
        return y[:, target_index] # (B,)

    ig = IntegratedGradients(forward_func)

    # compute attributions; optionally get convergence delta
    attr = ig.attribute(
        x,
        baselines=baseline,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=return_delta
    )

    if return_delta:
        attr, delta = attr

    # Aggregate: sum over time (T) -> per-feature, then mean over batch (B)
    if use_absolute:
        contrib = attr.abs().sum(dim=1).mean(dim=0)   # (F,)
    else:
        contrib = attr.sum(dim=1).mean(dim=0)         # signed (can cancel)

    series = pd.Series(contrib.detach().cpu().numpy(), index=feature_names)\
                .sort_values(ascending=False)

    if return_delta:
        # Captum delta is (B,) — average for a scalar indicator
        delta_mean = float(torch.as_tensor(delta).mean().detach().cpu().item())
        return series, delta_mean

    return series

def plot_feature_importance(
    series: pd.Series,
    fname: str = "figures/ig_feature_importance.png",
    topk: int = 20,
    title: str = "Integrated Gradients (top features)"
):
    # Matplotlib OO style (official best practice)
    top = series.head(topk)
    fig, ax = plt.subplots(figsize=(10, 4))
    top.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("importance (arb. units)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
