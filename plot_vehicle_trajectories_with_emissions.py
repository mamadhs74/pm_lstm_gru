#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import os
import re

CSV_PATH = "GothaerPlatz_017_vehicles.csv"

VEH_ID_COL = "NO"
LAT_COL = "LATWGS84"
LON_COL = "LONGWGS84"
EMI_COL = "EMISSIONSPM10TOTAL"

MIN_POINTS = 5
os.makedirs("figures", exist_ok=True)

print("Loading CSV…")
df = pd.read_csv(CSV_PATH, sep=r"[;,|\s]+", engine="python")
print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())

# Convert decimal commas to dots if present
for col in [LAT_COL, LON_COL, EMI_COL]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing coordinates
df = df.dropna(subset=[LAT_COL, LON_COL])

veh_ids = sorted(df[VEH_ID_COL].unique())

base_colors = list(cm.get_cmap("tab20").colors)
while len(base_colors) < len(veh_ids):
    base_colors += base_colors

plt.figure(figsize=(12, 8))
ax = plt.gca()

for idx, vid in enumerate(veh_ids):
    sub = df[df[VEH_ID_COL] == vid]
    if len(sub) < MIN_POINTS:
        continue

    xs = sub[LON_COL].values
    ys = sub[LAT_COL].values
    em = sub[EMI_COL].fillna(0).values

    em_norm = (em - em.min()) / (em.max() - em.min() + 1e-9)

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    r, g, b = base_colors[idx]
    colors = [(r * (0.3 + 0.7 * v), g * (0.3 + 0.7 * v), b * (0.3 + 0.7 * v)) for v in em_norm]

    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    plt.text(xs[0], ys[0], str(vid), fontsize=7, color=base_colors[idx])

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Vehicle Trajectories With Emission Intensity")
plt.grid(True)
plt.tight_layout()

out_file = f"figures/vehicle_trajectories_{EMI_COL}.png"
plt.savefig(out_file, dpi=300)
plt.show()

print("Saved:", out_file)
