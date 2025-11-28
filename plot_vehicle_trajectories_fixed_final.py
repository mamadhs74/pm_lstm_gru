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

# -----------------------------
# Universal coordinate parser
# -----------------------------
def parse_coord(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()

    s = s.replace(",", ".")

    # detect DMS: 52°11'34"
    dms = re.match(r"(\d+)[^\d]+(\d+)[^\d]+(\d+\.?\d*)", s)
    if dms:
        d, m, sec = map(float, dms.groups())
        return d + m/60 + sec/3600

    # detect plain number
    num = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if num:
        return float(num[0])

    return np.nan


# -----------------------------
# Load file (CORRECT separator!)
# -----------------------------
print("Loading CSV…")
df = pd.read_csv(CSV_PATH, sep=";", engine="python")
print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())

# -----------------------------
# Convert coordinates
# -----------------------------
df[LAT_COL] = df[LAT_COL].apply(parse_coord)
df[LON_COL] = df[LON_COL].apply(parse_coord)

# Drop invalid
df = df.dropna(subset=[LAT_COL, LON_COL])
print("After cleaning:", len(df))

# -----------------------------
# Parse emission column
# -----------------------------
df[EMI_COL] = (
    df[EMI_COL]
    .astype(str)
    .str.replace(",", ".")
    .str.extract(r"([-+]?\d*\.\d+|[-+]?\d+)")
)
df[EMI_COL] = pd.to_numeric(df[EMI_COL], errors="coerce").fillna(0)

# -----------------------------
# Plotting
# -----------------------------
veh_ids = sorted(df[VEH_ID_COL].unique())
plt.figure(figsize=(12, 8))
ax = plt.gca()

base_colors = list(cm.get_cmap("tab20").colors)

for idx, vid in enumerate(veh_ids):
    sub = df[df[VEH_ID_COL] == vid]
    if len(sub) < 5:
        continue

    xs = sub[LON_COL].values
    ys = sub[LAT_COL].values
    em = sub[EMI_COL].values

    em_norm = (em - em.min()) / (em.max() - em.min() + 1e-9)

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # base color for the vehicle
    r, g, b = base_colors[idx % len(base_colors)]
    colors = [(r*(0.3+0.7*v), g*(0.3+0.7*v), b*(0.3+0.7*v)) for v in em_norm]

    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    ax.text(xs[0], ys[0], str(vid), fontsize=6, color=(r, g, b))

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Vehicle Trajectories with Emission Intensity")
plt.grid(True)
plt.tight_layout()

out_file = "figures/vehicle_trajectories_FIXED_FINAL.png"
plt.savefig(out_file, dpi=300)
plt.show()

print("Saved:", out_file)
