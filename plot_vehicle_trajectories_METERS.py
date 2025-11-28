#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import os

CSV_PATH = "GothaerPlatz_017_vehicles.csv"

VEH_ID_COL = "NO"
LAT_COL = "LATWGS84"
LON_COL = "LONGWGS84"
EMI_COL = "EMISSIONSPM10TOTAL"

os.makedirs("figures", exist_ok=True)

# ---------------------------
# Load data with correct sep
# ---------------------------
df = pd.read_csv(CSV_PATH, sep=";")
print("Loaded rows:", len(df))

# ---------------------------
# Parse coordinates
# ---------------------------
def parse_coord(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    return float(nums[0]) if nums else np.nan

df[LAT_COL] = df[LAT_COL].apply(parse_coord)
df[LON_COL] = df[LON_COL].apply(parse_coord)

df = df.dropna(subset=[LAT_COL, LON_COL])
print("Remaining after coord clean:", len(df))

# ---------------------------
# Convert emission
# ---------------------------
df[EMI_COL] = (
    df[EMI_COL]
    .astype(str)
    .str.replace(",", ".")
    .str.extract(r"([-+]?\d*\.\d+|[-+]?\d+)")
)
df[EMI_COL] = pd.to_numeric(df[EMI_COL], errors="coerce").fillna(0)

# ---------------------------
# Convert GPS to meters
# ---------------------------
lat0 = df[LAT_COL].mean()
lon0 = df[LON_COL].mean()

df["x_m"] = (df[LON_COL] - lon0) * 111320 * np.cos(np.radians(lat0))
df["y_m"] = (df[LAT_COL] - lat0) * 110540

# ---------------------------
# Plot everything
# ---------------------------
veh_ids = sorted(df[VEH_ID_COL].unique())

plt.figure(figsize=(12, 9))
ax = plt.gca()

base_colors = list(cm.get_cmap("tab20").colors)

for idx, vid in enumerate(veh_ids):
    sub = df[df[VEH_ID_COL] == vid]
    if len(sub) < 5:
        continue

    xs = sub["x_m"].values
    ys = sub["y_m"].values
    em = sub[EMI_COL].values

    em_norm = (em - em.min()) / (em.max() - em.min() + 1e-9)

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    seg = np.concatenate([points[:-1], points[1:]], axis=1)

    r, g, b = base_colors[idx % len(base_colors)]
    colors = [(r*(0.3+0.7*v), g*(0.3+0.7*v), b*(0.3+0.7*v)) for v in em_norm]

    lc = LineCollection(seg, colors=colors, linewidths=2)
    ax.add_collection(lc)

    ax.text(xs[0], ys[0], str(vid), fontsize=7, color=(r, g, b))

plt.title("Vehicle Trajectories (meters) with PM10 Emission Intensity")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.grid(True)
plt.axis("equal")

out = "figures/vehicle_trajectories_PM10_meters.png"
plt.savefig(out, dpi=300)
plt.show()

print("Saved:", out)
