#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# === CONFIG ===
CSV_PATH = "data/GothaerPlatz_017_vehicles.csv"   # or the *_clean.csv later
SEP = ";"                                    # VERY IMPORTANT
VEH_ID_COL = "NO"                            # vehicle identifier
LON_COL = "LONGWGS84"
LAT_COL = "LATWGS84"

# which emission to visualize 
EMI_COL = "EMISSIONSPM10TOTAL_NUM"   


# EMI_COL = "EMISSIONSPM10BRAKES_NUM"

# EMI_COL = "EMISSIONSPM10TIRES_NUM"



MAX_VEHICLES = 15    # plot at most this many vehicles (for readability)
MIN_POINTS = 10      # skip vehicles shorter than this


def main():
    # 1) Load and basic clean
    df = pd.read_csv(CSV_PATH, sep=SEP)

    # Make sure required columns exist
    required = [VEH_ID_COL, LON_COL, LAT_COL, EMI_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Drop rows without coordinates
    df = df.dropna(subset=[LON_COL, LAT_COL])

    # Convert to float (in case they are strings)
    df[LON_COL] = df[LON_COL].astype(float)
    df[LAT_COL] = df[LAT_COL].astype(float)

    # 2) Pick some vehicles
    veh_ids = df[VEH_ID_COL].unique()
    print(f"Found {len(veh_ids)} vehicles.")
    veh_ids = veh_ids[:MAX_VEHICLES]  # limit for readability
    print(f"Plotting first {len(veh_ids)} vehicles:", veh_ids)

    # Base colormap for vehicle hues
    base_cmap = cm.get_cmap("tab20", len(veh_ids))

    plt.figure(figsize=(8, 8))

    # We will store one scatter handle to build a colorbar later
    example_scatter = None

    for idx, vid in enumerate(veh_ids):
        sub = df[df[VEH_ID_COL] == vid].copy()

        # optional: sort by simulation time if available
        if "SIMSEC_NUM" in sub.columns:
            sub = sub.sort_values("SIMSEC_NUM")

        # too few points? skip
        if len(sub) < MIN_POINTS:
            continue

        # emission for this vehicle
        emi = sub[EMI_COL].astype(float).values

        # if all NaN or constant -> just plot in light grey
        if np.all(np.isnan(emi)) or np.nanmax(emi) == np.nanmin(emi):
            plt.plot(sub[LON_COL], sub[LAT_COL],
                     color="lightgrey", linewidth=1, alpha=0.5)
            continue

        # normalize emissions to [0, 1] per vehicle
        emi_min = np.nanmin(emi)
        emi_max = np.nanmax(emi)
        emi_norm = (emi - emi_min) / (emi_max - emi_min + 1e-9)

        # base RGB color for this vehicle (different hue per vehicle)
        base_rgb = base_cmap(idx)  # (r,g,b,a)
        base_r, base_g, base_b = base_rgb[:3]

        # build per-point RGBA: same hue, alpha = f(emission)
        # low emission -> more transparent, high emission -> more opaque
        alphas = 0.2 + 0.8 * emi_norm  # in [0.2, 1.0]
        colors = np.zeros((len(sub), 4))
        colors[:, 0] = base_r
        colors[:, 1] = base_g
        colors[:, 2] = base_b
        colors[:, 3] = alphas

        sc = plt.scatter(
            sub[LON_COL], sub[LAT_COL],
            c=colors,
            s=10,
            marker="o",
            edgecolors="none",
            label=f"Vehicle {vid}" if idx < 5 else None,  # avoid legend overload
        )

        if example_scatter is None:
            example_scatter = sc

    plt.xlabel("Longitude (WGS84)")
    plt.ylabel("Latitude (WGS84)")
    plt.title(f"Gothaer Platz – per-vehicle trajectories\ncolor intensity ∝ {EMI_COL}")

    # Just a basic legend for first few vehicles
    plt.legend(loc="best", fontsize=8)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    import os
    os.makedirs("figures", exist_ok=True)
    out_path = f"figures/gothaer_trajectories_per_vehicle_{EMI_COL}.png"
    plt.savefig(out_path, dpi=300)
    print("Saved plot to:", out_path)

        # === simple per-vehicle totals for the same subset ===
    print("\n=== Computing total emissions per vehicle ===")

    totals = []
    for vid in veh_ids:
        sub = df[df[VEH_ID_COL] == vid]
        if len(sub) < MIN_POINTS:
            continue

        # convert emission column to float safely
        emi_vals = pd.to_numeric(sub[EMI_COL], errors='coerce').fillna(0)

        total_emi = emi_vals.sum()
        totals.append((vid, total_emi))

    # sort descending (highest emitters first)
    totals = sorted(totals, key=lambda x: x[1], reverse=True)

    if len(totals) == 0:
        print("No vehicles with emission data!")
        return

    # split into two lists
    vids_sorted = [v for v, _ in totals]
    vals_sorted = [v for _, v in totals]

    # ---- plot bar chart ---- #
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(vids_sorted)), vals_sorted, color="darkred")
    plt.xticks(range(len(vids_sorted)), vids_sorted, rotation=45)
    plt.ylabel(f"Total {EMI_COL}")
    plt.title(f"Per-Vehicle Total {EMI_COL}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # save
    import os
    os.makedirs("figures", exist_ok=True)
    out_bar = f"figures/gothaer_totals_{EMI_COL}.png"
    plt.savefig(out_bar, dpi=300)

    print("Saved per-vehicle totals plot to:", out_bar)

    plt.show()


if __name__ == "__main__":
    main()
