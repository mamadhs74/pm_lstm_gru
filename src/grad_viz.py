# src/grad_viz.py
import math
import torch
from torch.utils.tensorboard import SummaryWriter

def make_writer(logdir="runs", run_name="default"):
    """
    Create a TensorBoard writer.
    Open UI via: pixi run tensorboard
    """
    return SummaryWriter(log_dir=f"{logdir}/{run_name}")

def grad_flow(model):
    """
    Compute per-parameter gradient norms (L2) and a global norm.
    Returns (global_norm, list_of_tuples[(name, norm)]).
    """
    norms = []
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n = torch.linalg.vector_norm(g).item()
        norms.append((name, n))
        total_sq += n * n
    global_norm = math.sqrt(total_sq)
    return global_norm, norms

def log_gradients_tb(writer, model, step, tag_prefix="grads"):
    """
    Log gradient scalar norms and histograms to TensorBoard.
    See: torch.utils.tensorboard docs.
    """
    gnorm, norms = grad_flow(model)
    writer.add_scalar(f"{tag_prefix}/global_norm", gnorm, step)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # scalar: per-parameter L2 norm of grad
        writer.add_scalar(f"{tag_prefix}/param_norm/{name}", torch.linalg.vector_norm(p.grad).item(), step)
        # histogram: full gradient distribution
        writer.add_histogram(f"{tag_prefix}/hist/{name}", p.grad.detach().cpu(), step)
