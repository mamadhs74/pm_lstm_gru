import pandas as pd
import matplotlib.pyplot as plt

# Input files
booster_file = "data/processed/booster_clean.csv"
stopper_file = "data/processed/stopper_clean.csv"

# --------------------------------------------------------
# Load data
# --------------------------------------------------------
df_booster = pd.read_csv(booster_file)
df_stopper = pd.read_csv(stopper_file)

# --------------------------------------------------------
# Identify correct PM10 column
# --------------------------------------------------------
pm10_cols = [c for c in df_booster.columns if "PM10" in c]
print("Detected PM10 columns:", pm10_cols)

# Select PM10 mg/m3 column automatically
pm10_col = [c for c in pm10_cols if "mg" in c][0]
print("Using PM10 column:", pm10_col)

# --------------------------------------------------------
# Plot PM10 for both scenarios
# --------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(df_booster.index, df_booster[pm10_col], label="Booster", alpha=0.8)
plt.plot(df_stopper.index, df_stopper[pm10_col], label="Stopper", alpha=0.8)

plt.xlabel("Time (index)")
plt.ylabel("PM10 [mg/m³]")
plt.title("PM10 over Time — Booster vs Stopper")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/pm10_booster_vs_stopper.png", dpi=300)
plt.show()
