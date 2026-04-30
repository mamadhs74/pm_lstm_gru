import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "data/processed/test_predictions_p_va_physical.csv"
TRUE_COL = "P_VA_true"
PRED_COL = "P_VA_pred"
TIME_COL = "Time_since_midnight_UTC"

OUT_DIR = Path("ap6_ap7_outputs")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")

print("Columns:", df.columns.tolist())
print("Time range:", df[TIME_COL].min(), df[TIME_COL].max())

blocks = {
    "Block 1 - Beschleunigung": (8*3600 + 33*60, 8*3600 + 37*60),
    "Block 2 - Verzögerung": (8*3600 + 43*60, 8*3600 + 47*60),
    "Block 3 - Kurvenfahrt": (8*3600 + 51*60, 8*3600 + 56*60),
    "Block 9 - Cruise": (9*3600 + 26*60, 9*3600 + 32*60),
}

results = []

for name, (start, end) in blocks.items():
    block = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)].copy()

    if block.empty:
        print(f"WARNING: no data found for {name}")
        continue

    block["relative_time_s"] = block[TIME_COL] - start

    mae = (block[TRUE_COL] - block[PRED_COL]).abs().mean()
    rmse = ((block[TRUE_COL] - block[PRED_COL]) ** 2).mean() ** 0.5

    results.append({
        "Block": name,
        "Start_s": start,
        "End_s": end,
        "Mean_true": block[TRUE_COL].mean(),
        "Mean_pred": block[PRED_COL].mean(),
        "Max_true": block[TRUE_COL].max(),
        "Max_pred": block[PRED_COL].max(),
        "MAE": mae,
        "RMSE": rmse,
        "N_samples": len(block),
    })

    plt.figure(figsize=(10, 4))
    plt.plot(block[TIME_COL], block[TRUE_COL], label=f"{name} true")
    plt.plot(block[TIME_COL], block[PRED_COL], "--", label=f"{name} predicted")
    plt.title(name)
    plt.xlabel("Time within block [s]")
    plt.ylabel(TRUE_COL)
    plt.legend()
    plt.tight_layout()

    safe_name = name.replace(" ", "_").replace("-", "").replace("ä", "ae")
    plt.savefig(OUT_DIR / f"{safe_name}_true_vs_pred.png", dpi=300)
    plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "ap6_ap7_block_statistics.csv", index=False)

print("\nStatistics:")
print(results_df)
print("\nSaved outputs to:", OUT_DIR)