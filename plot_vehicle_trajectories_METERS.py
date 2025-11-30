#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import contextily as ctx
import os

CSV_PATH = "GothaerPlatz_017_vehicles.csv"

VEH_ID_COL = "NO"
LAT_COL = "LATWGS84"
LON_COL = "LONGWGS84"
EMI_COL = "EMISSIONSPM10TOTAL"

os.makedirs("figures", exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(CSV_PATH, sep=";")
print("Loaded rows:", len(df))

def parse_coord(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    return float(nums[0]) if nums else np.nan

df[LAT_COL] = df[LAT_COL].apply(parse_coord)
df[LON_COL] = df[LON_COL].apply(parse_coord)
df = df.dropna(subset=[LAT_COL, LON_COL])

# Emission fix
df[EMI_COL] = (
    df[EMI_COL].astype(str)
    .str.replace(",", ".")
    .str.extract(r"([-+]?\d*\.\d+|[-+]?\d+)")
)
df[EMI_COL] = pd.to_numeric(df[EMI_COL], errors="coerce").fillna(0)

# ---------------------------
# Convert GPS → Web Mercator (for OSM tiles)
# ---------------------------
import pyproj
proj = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)

df["x"], df["y"] = proj.transform(df[LON_COL].values, df[LAT_COL].values)

# Global emission min/max
global_min = df[EMI_COL].min()
global_max = df[EMI_COL].max()

print("Global emission range:", global_min, global_max)

veh_ids = sorted(df[VEH_ID_COL].unique())

plt.figure(figsize=(12, 10))
ax = plt.gca()

base_colors = list(cm.get_cmap("tab20").colors)

# Gamma for contrast enhancement
gamma = 0.35

for idx, vid in enumerate(veh_ids):
    sub = df[df[VEH_ID_COL] == vid]
    if len(sub) < 3:
        continue

    xs = sub["x"].values
    ys = sub["y"].values
    em = sub[EMI_COL].values

    # Global normalization
    em_norm = (em - global_min) / (global_max - global_min + 1e-9)

    # Gamma correction for visibility
    em_gamma = em_norm ** gamma

    # Prepare segments
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    seg = np.concatenate([points[:-1], points[1:]], axis=1)

    # Unique vehicle base color
    r, g, b = base_colors[idx % len(base_colors)]

    # Emission: darkness + line width
    colors = [(r * v, g * v, b * v) for v in em_gamma]
    linewidths = 1 + 4 * em_gamma  # thicker = more emissions

    lc = LineCollection(seg, colors=colors, linewidths=linewidths)
    ax.add_collection(lc)

    # Mark start
    ax.text(xs[0], ys[0], str(vid), fontsize=7, color=(r, g, b))

# Add background map
xmin, xmax = df["x"].min(), df["x"].max()
ymin, ymax = df["y"].min(), df["y"].max()
ax.set_xlim(xmin - 20, xmax + 20)
ax.set_ylim(ymin - 20, ymax + 20)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

plt.title(f"Vehicle Trajectories with PM10 Emission (Gamma={gamma})")
plt.xlabel("Web Mercator X")
plt.ylabel("Web Mercator Y")
plt.tight_layout()

out = "figures/vehicle_trajectories_PM10_GAMMA.png"
plt.savefig(out, dpi=300)
plt.show()

print("Saved:", out)
