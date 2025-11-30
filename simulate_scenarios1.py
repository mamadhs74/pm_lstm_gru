#!/usr/bin/env python3
"""
Simulate baseline vs 'informed drivers' scenarios on GothaerPlatz_017_vehicles.csv

Baseline:
    P_VA_base, P_HA_base  (tire & brake PM proxies) computed from SPEED_NUM, ACCELERATION_NUM.

Scenario:
    Some vehicles are 'informed' (know traffic light states) and avoid harsh braking.
    For those vehicles, we reduce brake- and tire-related PM during strong decelerations.

Outputs:
    - data/GothaerPlatz_017_with_scenarios.csv  (enriched dataset)
    - figures/gothaer_per_vehicle_PMHA_reduction.png
    - figures/gothaer_spatial_PMHA_reduction.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import re

# -----------------------------
# Paths / constants
# -----------------------------
CSV_PATH = "GothaerPlatz_017_vehicles.csv"
OUT_CSV  = "data/GothaerPlatz_017_with_scenarios.csv"

VEH_ID_COL = "NO"
TIME_COL   = "SIMSEC_NUM"          # simulation time [s]
V_COL      = "SPEED_NUM"           # km/h
A_COL      = "ACCELERATION_NUM"    # m/s^2
LAT_COL    = "LATWGS84"
LON_COL    = "LONGWGS84"

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def parse_coord(val):
    """Parse LAT/LON (with commas, units, etc.) into float degrees."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    return float(nums[0]) if nums else np.nan


# -----------------------------
# 1. Load & basic cleaning
# -----------------------------
print("Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, sep=";")
print("Loaded rows:", len(df))

required_cols = [VEH_ID_COL, TIME_COL, V_COL, A_COL, LAT_COL, LON_COL]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# Ensure numeric speed/acc
df[V_COL] = pd.to_numeric(df[V_COL], errors="coerce")
df[A_COL] = pd.to_numeric(df[A_COL], errors="coerce")

# Drop rows without kinematics
df = df.dropna(subset=[V_COL, A_COL])
print("Remaining after dropping NaNs in V/A:", len(df))

# Parse coordinates
df[LAT_COL] = df[LAT_COL].apply(parse_coord)
df[LON_COL] = df[LON_COL].apply(parse_coord)
df = df.dropna(subset=[LAT_COL, LON_COL])
print("Remaining after coord clean:", len(df))

# Local meter coordinates (for spatial plots)
lat0 = df[LAT_COL].mean()
lon0 = df[LON_COL].mean()
df["x_m"] = (df[LON_COL] - lon0) * 111320 * np.cos(np.radians(lat0))
df["y_m"] = (df[LAT_COL] - lat0) * 110540

# -----------------------------
# 2. Define baseline PM proxies
# -----------------------------
v_ms  = df[V_COL].values / 3.6     # convert km/h → m/s
a     = df[A_COL].values          # m/s^2
a_pos = np.maximum(a, 0.0)
a_neg = np.minimum(a, 0.0)

# Simple surrogate models (dimensionless but consistent)
# Think of them as "relative PM intensity".
k_va = 1.0
k_ha = 1.0

P_VA_base = k_va * (v_ms**2) * (0.1 + a_pos)          # tyre: speed^2 + accel
P_HA_base = k_ha * (v_ms) * np.maximum(-a_neg, 0.0)   # brake: speed * decel

df["P_VA_base"] = P_VA_base
df["P_HA_base"] = P_HA_base

print("Baseline PM stats:")
print("  P_VA_base:", df["P_VA_base"].describe())
print("  P_HA_base:", df["P_HA_base"].describe())

# -----------------------------
# 3. Define informed vehicles & scenario PM
# -----------------------------
veh_ids = sorted(df[VEH_ID_COL].unique())
print("Number of vehicles:", len(veh_ids))

# Simple rule: every 2nd vehicle is 'informed'
informed_ids = set(veh_ids[::2])
df["informed"] = df[VEH_ID_COL].isin(informed_ids).astype(int)

print("Number of informed vehicles:", len(informed_ids))

# Scenario = copy baseline
df["P_VA_scen"] = df["P_VA_base"].values
df["P_HA_scen"] = df["P_HA_base"].values

# For informed vehicles only, reduce PM during harsh decel
mask_informed = df["informed"] == 1
mask_harsh_decel = (a_neg < -1.0) & (v_ms > 10.0)   # strong brake at >10 km/h

mask = mask_informed & mask_harsh_decel

# Parameters: how much we reduce
brake_reduction_factor = 0.6   # 40% reduction in brake PM
tyre_reduction_factor  = 0.9   # 10% reduction in tyre PM (less stop-go)

df.loc[mask, "P_HA_scen"] = df.loc[mask, "P_HA_base"] * brake_reduction_factor
df.loc[mask, "P_VA_scen"] = df.loc[mask, "P_VA_base"] * tyre_reduction_factor

# -----------------------------
# 4. Aggregate per vehicle
# -----------------------------
agg = (
    df.groupby(VEH_ID_COL)[["P_VA_base", "P_VA_scen", "P_HA_base", "P_HA_scen"]]
    .sum()
    .reset_index()
)

agg["P_VA_reduction"] = agg["P_VA_base"] - agg["P_VA_scen"]
agg["P_HA_reduction"] = agg["P_HA_base"] - agg["P_HA_scen"]

# Global totals
total_base_VA = agg["P_VA_base"].sum()
total_scen_VA = agg["P_VA_scen"].sum()
total_base_HA = agg["P_HA_base"].sum()
total_scen_HA = agg["P_HA_scen"].sum()

print("\n=== Global totals (sum over all vehicles) ===")
print(f"Tire PM proxy  baseline: {total_base_VA:.2f}  scenario: {total_scen_VA:.2f}  "
      f"Δ = {total_base_VA - total_scen_VA:.2f}  ({100*(1 - total_scen_VA/total_base_VA):.1f}% reduction)")

print(f"Brake PM proxy baseline: {total_base_HA:.2f}  scenario: {total_scen_HA:.2f}  "
      f"Δ = {total_base_HA - total_scen_HA:.2f}  ({100*(1 - total_scen_HA/total_base_HA):.1f}% reduction)")

# Save full enriched dataset
df.to_csv(OUT_CSV, index=False)
print("Saved enriched scenario dataset to:", OUT_CSV)

# -----------------------------
# 5. Plot 1 – Per-vehicle brake PM reduction
# -----------------------------
agg_sorted = agg.sort_values("P_HA_base", ascending=False).reset_index(drop=True)

plt.figure(figsize=(10, 5))
x = np.arange(len(agg_sorted))

plt.bar(x - 0.2, agg_sorted["P_HA_base"], width=0.4, label="Baseline brake PM (P_HA_base)")
plt.bar(x + 0.2, agg_sorted["P_HA_scen"], width=0.4, label="Scenario brake PM (P_HA_scen)")

plt.xticks(x, agg_sorted[VEH_ID_COL], rotation=45, fontsize=7)
plt.ylabel("Brake PM proxy (sum over trajectory)")
plt.xlabel("Vehicle ID")
plt.title("Per-vehicle brake PM – baseline vs informed scenario")
plt.legend()
plt.tight_layout()

out_bar = "figures/gothaer_per_vehicle_PMHA_reduction.png"
plt.savefig(out_bar, dpi=300)
plt.show()
print("Saved per-vehicle PM plot to:", out_bar)

# -----------------------------
# 6. Plot 2 – Spatial map of brake PM *reduction*
# -----------------------------
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Normalise reduction for colour mapping
df["P_HA_delta"] = df["P_HA_base"] - df["P_HA_scen"]
delta = df["P_HA_delta"].values
delta_norm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-9)

veh_ids = sorted(df[VEH_ID_COL].unique())
base_colors = list(cm.get_cmap("tab20").colors)

for idx, vid in enumerate(veh_ids):
    sub = df[df[VEH_ID_COL] == vid]
    if len(sub) < 5:
        continue

    xs = sub["x_m"].values
    ys = sub["y_m"].values
    d  = sub["P_HA_delta"].values

    # normalise per-vehicle delta to highlight where reduction happens
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    seg = np.concatenate([points[:-1], points[1:]], axis=1)

    r, g, b = base_colors[idx % len(base_colors)]
    colors = [(r * (0.3 + 0.7 * v), g * (0.3 + 0.7 * v), b * (0.3 + 0.7 * v)) for v in d_norm]
    widths = 0.5 + 4.0 * d_norm

    lc = LineCollection(seg, colors=colors, linewidths=widths, alpha=0.9)
    ax.add_collection(lc)

plt.title("Spatial distribution of brake PM reduction (baseline − scenario)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.axis("equal")

out_spatial = "figures/gothaer_spatial_PMHA_reduction.png"
plt.savefig(out_spatial, dpi=300)
plt.show()
print("Saved spatial PM reduction plot to:", out_spatial)
