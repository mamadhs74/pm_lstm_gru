import torch
import torch.nn.functional as F

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

class HuberLoss(torch.nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, y_pred, y_true):
        return F.huber_loss(y_pred, y_true, delta=self.delta)
