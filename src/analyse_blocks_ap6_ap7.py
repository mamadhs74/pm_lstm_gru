import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "data/processed/1_clean_decimal_fixed.csv"
PRED_COL = "P_VA"
TIME_COL = "Zeitstempel"

OUT_DIR = Path("ap6_ap7_outputs")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="%H:%M:%S")

blocks = {
    "Block 1 - Beschleunigung": ("10:33:00", "10:37:00"),
    "Block 2 - Verzögerung": ("10:43:00", "10:47:00"),
    "Block 3 - Kurvenfahrt": ("10:51:00", "10:56:00"),
    "Block 9 - Cruise": ("11:26:00", "11:32:00"),
}

results = []

plt.figure(figsize=(12, 5))

for name, (start, end) in blocks.items():
    start = pd.to_datetime(start, format="%H:%M:%S")
    end = pd.to_datetime(end, format="%H:%M:%S")

    block = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)].copy()

    if block.empty:
        print(f"WARNING: no data found for {name}")
        continue

    results.append({
        "Block": name,
        "Start": start.strftime("%H:%M:%S"),
        "End": end.strftime("%H:%M:%S"),
        "Mean": block[PRED_COL].mean(),
        "Max": block[PRED_COL].max(),
        "Std": block[PRED_COL].std(),
        "N_samples": len(block),
    })

    plt.plot(block[TIME_COL], block[PRED_COL], label=name)

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "ap6_ap7_block_statistics.csv", index=False)

plt.title("Particle-emission response across driving scenarios")
plt.xlabel("Time")
plt.ylabel(PRED_COL)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "ap6_ap7_block_plot.png", dpi=300)

print("\nSaved:")
print(OUT_DIR / "ap6_ap7_block_statistics.csv")
print(OUT_DIR / "ap6_ap7_block_plot.png")
print("\nStatistics:")
print(results_df)