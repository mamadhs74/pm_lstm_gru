#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import pickle
import os

from src.models import TCNRegressor
from torch.optim import AdamW
from torch.nn import SmoothL1Loss
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

CSV = "GothaerPlatz_017_vehicles.csv"

FEATURES = [
    "SPEED_NUM",
    "ACCELERATION_NUM",
    "LANE_LINK_GRADIENT_NUM",
    "FUELCONSUMPTION_NUM"
]

TARGETS = [
    "EMISSIONSPM10TOTAL_NUM",
    "EMISSIONSPM10BRAKES_NUM",
    "EMISSIONSPM10TIRES_NUM"
]

# -------------------------------
# Dataset class
# -------------------------------
class GothaerDataset(Dataset):
    def __init__(self, df, feats, targs):
        self.X = df[feats].values.astype(np.float32)
        self.Y = df[targs].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -------------------------------
# Load + clean
# -------------------------------
print("Loading CSV…")
df = pd.read_csv(CSV, sep=";")

# Convert commas
for col in FEATURES + TARGETS:
    df[col] = (
        df[col].astype(str)
        .str.replace(",", ".")
        .str.extract(r"([-+]?\d*\.\d+|[-+]?\d+)")
    )
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# -------------------------------
# Train/Val split
# -------------------------------
N = len(df)
train_df = df.iloc[: int(N*0.8)]
val_df   = df.iloc[int(N*0.8):]

train_ds = GothaerDataset(train_df, FEATURES, TARGETS)
val_ds   = GothaerDataset(val_df, FEATURES, TARGETS)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)

# -------------------------------
# Model
# -------------------------------
model = TCNRegressor(
    input_size=len(FEATURES),
    out_dim=len(TARGETS),
    channels_list=(32, 32, 32),
    kernel_size=3,
    dropout=0.1
)

opt = AdamW(model.parameters(), lr=1e-3)
loss_fn = SmoothL1Loss()

# -------------------------------
# Training
# -------------------------------
print("Training...")
train_losses = []
val_losses   = []

for epoch in range(40):
    model.train()
    t_loss = 0

    for xb, yb in train_loader:
        opt.zero_grad()
        pred = model(xb.unsqueeze(1))
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        t_loss += loss.item()

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb.unsqueeze(1))
            v_loss += loss_fn(pred, yb).item()

    train_losses.append(t_loss)
    val_losses.append(v_loss)

    print(f"Epoch {epoch:03d} | Train {t_loss:.4f} | Val {v_loss:.4f}")

torch.save(model.state_dict(), ".runs/gothaer_tcn.pt")
print("Saved model to .runs/gothaer_tcn.pt")

plt.plot(train_losses, label="train")
plt.plot(val_losses,   label="val")
plt.legend()
plt.savefig("figures/gothaer_training_curve.png")
