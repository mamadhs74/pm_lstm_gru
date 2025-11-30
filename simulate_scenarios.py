# simulate_scenarios.py
import pandas as pd
import numpy as np
from gothaer_preprocess import load_gothaer

VEHICLES = [1,2,3,4,5]

def smooth_speed(v, window=15):
    """Simple eco-driving smoothing filter."""
    return np.convolve(v, np.ones(window)/window, mode="same")

def generate_scenarios():
    df = load_gothaer()

    baseline = df[df["NO"].isin(VEHICLES)].copy()
    informed = baseline.copy()

    # INFORMED scenario: smooth speed for each vehicle
    for vid in VEHICLES:
        sub = informed[informed["NO"] == vid]
        idxs = sub.index
        v = sub["SPEED_NUM"].values
        v_smooth = smooth_speed(v, window=25)
        informed.loc[idxs, "SPEED_NUM"] = v_smooth

        # recompute acceleration
        informed.loc[idxs, "ACCELERATION_NUM"] = np.gradient(v_smooth)

    baseline.to_csv("baseline_5cars.csv", index=False)
    informed.to_csv("informed_5cars.csv", index=False)
    print("Saved baseline_5cars.csv and informed_5cars.csv")

if __name__ == "__main__":
    generate_scenarios()
