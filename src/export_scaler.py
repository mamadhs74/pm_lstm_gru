# src/export_scaler.py
import joblib
import pandas as pd

from src.config import cfg, resolved_feature_cols, resolved_target_cols
from src.preprocess import prepare_frames

def main():
    # ensure config points to 1_clean, same as training
    raw = pd.read_csv("data/processed/1_clean.csv")
    
    # prepare_frames SHOULD return (splits, scaler) — if it doesn't,
    # you may need to modify it to do so. Conceptually:
    splits, scaler = prepare_frames(raw, return_scaler=True)
    
    joblib.dump(scaler, "artifacts/scaler_1clean.pkl")
    print("saved scaler to artifacts/scaler_1clean.pkl")
    print("feature cols:", resolved_feature_cols(cfg))
    print("target cols:", resolved_target_cols(cfg))

if __name__ == "__main__":
    main()
