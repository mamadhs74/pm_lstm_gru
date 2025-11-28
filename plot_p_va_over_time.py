# plot_p_va_over_time.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "data/processed/1_clean_decimal_fixed.csv"   # change if needed
P_VA_COL = "P_VA"  # <-- set exact column name here

def main():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # ----------------------------------------
    # FIND THE CORRECT TIME COLUMN
    # ----------------------------------------
    time_candidates = ["time_s", "X_Achse_[s]", "timestamp", "Zeitstempel"]

    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        print("No explicit time column found — using index instead.")
        t = df.index
        t_label = "sample index"
    else:
        t = df[time_col].astype(float)
        t_label = time_col
        print(f"Using time column: {time_col}")

    # ----------------------------------------
    # READ P_VA column
    # ----------------------------------------
    if P_VA_COL not in df.columns:
        raise KeyError(
            f"Column '{P_VA_COL}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    y = df[P_VA_COL].astype(float)

    # ----------------------------------------
    # PLOT
    # ----------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(t, y, label=P_VA_COL, linewidth=2)
    plt.title(f"{P_VA_COL} over time")
    plt.xlabel(t_label)
    plt.ylabel(P_VA_COL)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = Path("figures") / f"{P_VA_COL}_over_time.png"
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(out_path)
    plt.show()

    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    main()
