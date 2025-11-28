#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "GothaerPlatz_017_vehicles.csv"

LAT_COL = "LATWGS84"
LON_COL = "LONGWGS84"
SPEED_COL = "SPEED"
ACC_COL = "ACCELERATION"
EMI_COL = "EMISSIONSPM10TOTAL"

os.makedirs("figures", exist_ok=True)

print("\n=== Loading CSV ===")
df = pd.read_csv(CSV_PATH, sep=None, engine="python")
print("Rows loaded:", len(df))

# -------------------------------------
# FIX DECIMAL COMMAS IN ALL COLUMNS
# -------------------------------------
def fix_commas(x):
    if isinstance(x, str):
        return x.replace(",", ".")
    return x

df = df.applymap(fix_commas)

# -------------------------------------
# CONVERT ALL POSSIBLE NUMERIC COLUMNS
# -------------------------------------
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    except:
        pass

# -------------------------------------
# Print diagnostics
# -------------------------------------
print("\n=== Column types ===")
print(df.dtypes)

print("\n=== Example values ===")
print(df[[SPEED_COL, ACC_COL, EMI_COL]].head())

# -------------------------------------
# Remove rows with invalid numbers
# -------------------------------------
df = df[
    df[SPEED_COL].apply(lambda x: isinstance(x, (int, float))) &
    df[ACC_COL].apply(lambda x: isinstance(x, (int, float))) &
    df[EMI_COL].apply(lambda x: isinstance(x, (int, float)))
]

print("\nValid rows after cleaning:", len(df))

# ======================================================================
# PLOTTING
# ======================================================================

# Speed vs Emission
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[SPEED_COL], y=df[EMI_COL], s=8)
plt.xlabel("Speed [km/h]")
plt.ylabel("PM10 Emission [mg/s]")
plt.title("Emission vs Speed")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/emission_vs_speed.png", dpi=200)
plt.close()

# Acceleration vs Emission
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[ACC_COL], y=df[EMI_COL], s=8)
plt.xlabel("Acceleration [m/s²]")
plt.ylabel("PM10 Emission [mg/s]")
plt.title("Emission vs Acceleration")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/emission_vs_acc.png", dpi=200)
plt.close()

print("\nSaved plots:")
print(" - figures/emission_vs_speed.png")
print(" - figures/emission_vs_acc.png")
print("\nDone.\n")
