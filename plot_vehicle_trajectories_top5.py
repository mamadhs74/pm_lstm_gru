#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import os

CSV_PATH = "GothaerPlatz_017_vehicles.csv"

VEH_ID_COL = "NO"
LAT_COL = "LATWGS84"
LON_COL = "LONGWGS84"
EMI_COL = "EMISSIONSPM10TOTAL"   # can switch to BRAKES / TIRES later

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
# Convert emission to float
# ---------------------------
df[EMI_COL] = (
    df[EMI_COL]
    .astype(str)
    .str.replace(",", ".")
    .str.extract(r"([-+]?\d*\.\d+|[-+]?\d+)")
)
df[EMI_COL] = pd.to_numeric(df[EMI_COL], errors="coerce").fillna(0.0)

# ---------------------------
# Convert GPS to local meters
# ---------------------------
lat0 = df[LAT_COL].mean()
lon0 = df[LON_COL].mean()

df["x_m"] = (df[LON_COL] - lon0) * 111320 * np.cos(np.radians(lat0))
df["y_m"] = (df[LAT_COL] - lat0) * 110540

# ---------------------------
# Select only first 5 vehicles
# ---------------------------
veh_ids_all = sorted(df[VEH_ID_COL].unique())
veh_ids = veh_ids_all[:5]
print("Plotting vehicles:", veh_ids)

# global emission range (for consistent darkness across cars)
emi_global_min = df[EMI_COL].min()
emi_global_max = df[EMI_COL].max()
emi_span = emi_global_max - emi_global_min if emi_global_max > emi_global_min else 1.0

# threshold for hotspot overlay (top 10% emissions)
emi_hot_thr = df[EMI_COL].quantile(0.90)


plt.figure(figsize=(12, 9))
ax = plt.gca()

# distinct base colors for up to 5 vehicles
base_colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(veh_ids)))

legend_proxies = []

for idx, vid in enumerate(veh_ids):
    sub = df[df[VEH_ID_COL] == vid]
    if len(sub) < 5:
        continue

    xs = sub["x_m"].values
    ys = sub["y_m"].values
    em = sub[EMI_COL].values

    # normalize globally, NOT per-vehicle
    em_norm = (em - emi_global_min) / emi_span
    # gamma to boost contrast
    em_norm_gamma = np.power(em_norm, 0.5)

    # prepare line segments
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    r, g, b, _ = base_colors[idx]
    # alpha + linewidth depend on emission
    colors = [(r, g, b, 0.15 + 0.85 * v) for v in em_norm_gamma]
    widths = 1.0 + 3.0 * em_norm_gamma

    lc = LineCollection(segments, colors=colors, linewidths=widths)
    ax.add_collection(lc)

    # label first point with vehicle ID
    ax.text(xs[0], ys[0], str(vid), fontsize=8, color=(r, g, b))

    # for legend
    legend_proxies.append(
        Line2D([0], [0], color=(r, g, b), lw=3, label=f"Vehicle {vid}")
    )

# ---------------------------
# Hotspot overlay (top 10% emissions)
# ---------------------------
hot = df[df[EMI_COL] >= emi_hot_thr]
sc = ax.scatter(
    hot["x_m"],
    hot["y_m"],
    c=hot[EMI_COL],
    cmap="inferno",
    s=15,
    alpha=0.9,
    edgecolor="none",
    label="Top 10% emission points",
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(f"{EMI_COL} (high = dark)")

# legend for vehicles
legend_proxies.append(
    Line2D([], [], linestyle="none", marker="o", color="black", label="Emission hotspot")
)
ax.legend(handles=legend_proxies, loc="upper right", fontsize=8)

plt.title("Top 5 Vehicle Trajectories with PM10 Emission Intensity")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.axis("equal")

out = "figures/vehicle_trajectories_PM10_top5_hotspots.png"
plt.savefig(out, dpi=300)
plt.show()

print("Saved:", out)
