#!/usr/bin/env python3
import pandas as pd

input_path  = "data/processed/1_clean.csv"
output_path = "data/processed/1_clean_decimal_fixed.csv"

print("Loading:", input_path)
df = pd.read_csv(input_path, dtype=str)  # load everything as string

# Replace decimal commas with dots
df = df.replace(",", ".", regex=True)

# Now convert all columns that CAN be numbers
for col in df.columns:
    try:
        df[col] = df[col].astype(float)
    except Exception:
        # columns that cannot be numeric just stay as strings
        pass

# Save cleaned CSV
df.to_csv(output_path, index=False)
print("Saved cleaned CSV to:", output_path)
