#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

CSV_PATH = "GothaerPlatz_017_vehicles.csv"
os.makedirs("figures", exist_ok=True)

# ---------------------------
# 1. Load & basic cleaning
# ---------------------------
print("Loading CSV…")
df = pd.read_csv(CSV_PATH, sep=";")
print("Loaded rows:", len(df))

# We will use ONLY vehicles 1–5
veh_ids = [1, 2, 3, 4, 5]
df = df[df["NO"].isin(veh_ids)].copy()
print("Rows for vehicles 1–5:", len(df))

# Ensure numeric columns we care about are float
# The *_NUM columns in your file are already numeric, but we coerce to be safe.
numeric_cols = [
    "SPEED_NUM",
    "ACCELERATION_NUM",
    "LANE_LINK_GRADIENT_NUM",
    "FUELCONSUMPTION_NUM",
    "EMISSIONSPM10TOTAL_NUM",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values in these core columns
df = df.dropna(subset=numeric_cols)
print("Rows after dropping NaNs:", len(df))

# ---------------------------
# 2. Define features & target
# ---------------------------
FEATURES = [
    "SPEED_NUM",
    "ACCELERATION_NUM",
    "LANE_LINK_GRADIENT_NUM",
    "FUELCONSUMPTION_NUM",
]
TARGET = "EMISSIONSPM10TOTAL_NUM"

X = df[FEATURES].values
y = df[TARGET].values

# ---------------------------
# 3. Fit baseline emission model
# ---------------------------
print("Fitting RandomForestRegressor for emission model…")
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
)
rf.fit(X, y)
print("Done.")

# Predict baseline emissions (for sanity)
df["EMI_BASE_PRED"] = rf.predict(X)

# ---------------------------
# 4. Build eco / informed-driver scenario
#    Simple heuristic:
#    - smooth speed with rolling mean per vehicle
#    - recompute acceleration from smoothed speed
#    - reuse other features
# ---------------------------

# First, make sure SIMSEC_NUM is numeric and sort by time
df["SIMSEC_NUM"] = pd.to_numeric(df["SIMSEC_NUM"], errors="coerce")
df = df.dropna(subset=["SIMSEC_NUM"])
df = df.sort_values(["NO", "SIMSEC_NUM"]).reset_index(drop=True)

# We'll assume Δt ≈ 0.1 s based on 18.30, 18.40, …
DT = 0.1

def build_eco_profile(group: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Given rows for a single vehicle (sorted by SIMSEC_NUM),
    create an eco version:
      - SPEED_ECO: rolling mean smoothing
      - ACC_ECO: finite-difference of SPEED_ECO / dt
    """
    g = group.copy()

    # Smooth speed (moving average in time)
    g["SPEED_ECO"] = g["SPEED_NUM"].rolling(window=window_size, center=True, min_periods=1).mean()

    # Compute eco acceleration from smoothed speed
    # Convert km/h -> m/s for diff, then back
    speed_ms = g["SPEED_ECO"] * (1000.0 / 3600.0)
    acc_ms2 = speed_ms.diff() / DT
    acc_ms2.iloc[0] = 0.0  # first sample, assume 0
    g["ACC_ECO"] = acc_ms2

    return g

# Apply eco profile per vehicle
groups = []
for vid, sub in df.groupby("NO"):
    sub_sorted = sub.sort_values("SIMSEC_NUM")
    eco = build_eco_profile(sub_sorted, window_size=7)
    groups.append(eco)

df_eco = pd.concat(groups, ignore_index=True)

# Now predict eco emissions using the same RF model
X_eco = df_eco.copy()
X_eco["ACCEL_ECO_FOR_MODEL"] = X_eco["ACC_ECO"]

# Keep other features same as baseline, except we swap acceleration
X_eco_model = np.column_stack([
    X_eco["SPEED_ECO"].values,
    X_eco["ACCEL_ECO_FOR_MODEL"].values,
    X_eco["LANE_LINK_GRADIENT_NUM"].values,
    X_eco["FUELCONSUMPTION_NUM"].values,
])

df_eco["EMI_ECO_PRED"] = rf.predict(X_eco_model)

# ---------------------------
# 5. Aggregate per-vehicle metrics
#    - total emissions baseline & eco
#    - travel time baseline & eco
# ---------------------------
agg_rows = []

for vid, sub_b in df.groupby("NO"):
    sub_e = df_eco[df_eco["NO"] == vid]

    # Baseline
    t_min_b = sub_b["SIMSEC_NUM"].min()
    t_max_b = sub_b["SIMSEC_NUM"].max()
    travel_time_b = t_max_b - t_min_b

    emi_b = sub_b["EMI_BASE_PRED"].sum()

    # Eco
    t_min_e = sub_e["SIMSEC_NUM"].min()
    t_max_e = sub_e["SIMSEC_NUM"].max()
    travel_time_e = t_max_e - t_min_e

    emi_e = sub_e["EMI_ECO_PRED"].sum()

    agg_rows.append({
        "NO": vid,
        "travel_time_baseline": travel_time_b,
        "travel_time_eco": travel_time_e,
        "emi_baseline": emi_b,
        "emi_eco": emi_e,
    })

agg = pd.DataFrame(agg_rows).sort_values("NO")
print("\nPer-vehicle summary (baseline vs eco):")
print(agg.to_string(index=False))

agg.to_csv("figures/scenario_5cars_summary.csv", index=False)
print("Saved per-vehicle summary to figures/scenario_5cars_summary.csv")

# ---------------------------
# 6. Plot 1: per-vehicle total emissions (bar chart)
# ---------------------------
plt.figure(figsize=(10, 6))
x = np.arange(len(agg["NO"]))
width = 0.35

plt.bar(x - width/2, agg["emi_baseline"], width, label="Baseline")
plt.bar(x + width/2, agg["emi_eco"], width, label="Eco / informed")

for i, (b, e) in enumerate(zip(agg["emi_baseline"], agg["emi_eco"])):
    if b > 0:
        reduction = 100.0 * (b - e) / b
        plt.text(x[i], max(b, e) * 1.01, f"{reduction:.1f}%", ha="center", va="bottom", fontsize=8)

plt.xticks(x, agg["NO"])
plt.xlabel("Vehicle ID")
plt.ylabel("Sum of PM10 emissions (model units)")
plt.title("Total PM10 Emissions per Vehicle: Baseline vs Eco (5 cars)")
plt.legend()
plt.tight_layout()

out_bar = "figures/scenario_5cars_emissions_bar.png"
plt.savefig(out_bar, dpi=300)
plt.close()
print("Saved:", out_bar)

# ---------------------------
# 7. Plot 2: travel time vs emissions (scatter with arrows)
# ---------------------------
plt.figure(figsize=(10, 6))

for _, row in agg.iterrows():
    vid = int(row["NO"])
    tb = row["travel_time_baseline"]
    te = row["travel_time_eco"]
    eb = row["emi_baseline"]
    ee = row["emi_eco"]

    # arrow from baseline to eco
    plt.scatter(tb, eb, color="tab:blue")
    plt.scatter(te, ee, color="tab:orange")
    plt.annotate(
        str(vid),
        (tb, eb),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
        color="black",
    )
    plt.arrow(
        tb,
        eb,
        te - tb,
        ee - eb,
        length_includes_head=True,
        head_width=0.02 * max(agg["emi_baseline"].max(), agg["emi_eco"].max()),
        head_length=0.02 * agg["travel_time_baseline"].max(),
        color="gray",
        alpha=0.7,
    )

plt.xlabel("Travel time [s]")
plt.ylabel("Total PM10 emissions (model units)")
plt.title("Travel Time vs Emissions per Vehicle: Baseline → Eco")
plt.grid(True)
plt.tight_layout()

out_scatter = "figures/scenario_5cars_time_vs_emissions.png"
plt.savefig(out_scatter, dpi=300)
plt.close()
print("Saved:", out_scatter)

# ---------------------------
# 8. Emission-cause analysis: feature importance
# ---------------------------
print("\n=== Emission cause analysis (RandomForest feature importances) ===")
importances = rf.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    print(f"{feat:25s}: {imp:.3f}")

plt.figure(figsize=(8, 4))
order = np.argsort(importances)[::-1]
plt.bar(np.array(FEATURES)[order], importances[order])
plt.ylabel("Importance")
plt.title("RandomForest Feature Importances for PM10 Emissions")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
out_imp = "figures/scenario_5cars_feature_importances.png"
plt.savefig(out_imp, dpi=300)
plt.close()
print("Saved feature importance plot to:", out_imp)

print("\nDone.")
