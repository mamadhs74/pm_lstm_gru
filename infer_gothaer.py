#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
from src.models import TCNRegressor
import matplotlib.pyplot as plt

FEATURES = ["SPEED_NUM","ACCELERATION_NUM","LANE_LINK_GRADIENT_NUM","FUELCONSUMPTION_NUM"]
TARGETS  = ["EMISSIONSPM10TOTAL_NUM","EMISSIONSPM10BRAKES_NUM","EMISSIONSPM10TIRES_NUM"]

def infer(csv):
    df = pd.read_csv(csv)
    for col in FEATURES + TARGETS:
        df[col] = (
            df[col].astype(str).str.replace(",",".")
            .str.extract(r"([-+]?\d*\.\d+|[-+]?\d+)")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    model = TCNRegressor(
        input_size=len(FEATURES),
        out_dim=len(TARGETS),
        channels_list=(32,32,32),
        kernel_size=3,
        dropout=0.1
    )
    model.load_state_dict(torch.load(".runs/gothaer_tcn.pt", map_location="cpu"))
    model.eval()

    X = df[FEATURES].values.astype(np.float32)
    preds = []

    with torch.no_grad():
        for x in X:
            y = model(torch.tensor(x).unsqueeze(0).unsqueeze(0)).numpy()
            preds.append(y.squeeze())

    preds = np.array(preds)
    df["PM10_pred"] = preds[:,0]
    df.to_csv(csv.replace(".csv","_pred.csv"), index=False)

    plt.plot(df["PM10_pred"])
    plt.title(csv)
    plt.savefig(csv.replace(".csv","_plot.png"))
    plt.close()

infer("baseline_5cars.csv")
infer("informed_5cars.csv")
