# infer_emissions.py
import torch
import numpy as np
import pandas as pd
import pickle

from gothaer_preprocess import load_gothaer
from src.models import TCNRegressor

FEATURES = ["SPEED_NUM", "ACCELERATION_NUM"]   # extend if needed
TARGETS = ["PM10_total"]  # or your model output names

CKPT = ".runs/tcn_best.pt"
FSC = ".runs/feature_scaler.pkl"
TSC = ".runs/target_scaler.pkl"

def load_model():
    with open(FSC, "rb") as f: fsc = pickle.load(f)
    with open(TSC, "rb") as f: tsc = pickle.load(f)

    model = TCNRegressor(
        input_size=len(FEATURES),
        out_dim=len(TARGETS),
        channels_list=[64,64,64],
        kernel_size=3,
        dropout=0.3,
    )
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    return model, fsc, tsc

def predict_file(csv_path):
    df = pd.read_csv(csv_path)
    model, fsc, tsc = load_model()

    X = fsc.transform(df[FEATURES])
    preds = []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(0)
            y = model(x).cpu().numpy()[0]
            preds.append(y)

    preds = np.array(preds)
    preds_real = tsc.inverse_transform(preds)

    for i,t in enumerate(TARGETS):
        df[t+"_pred"] = preds_real[:,i]

    out = csv_path.replace(".csv", "_with_emissions.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    predict_file("baseline_5cars.csv")
    predict_file("informed_5cars.csv")
