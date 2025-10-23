# scripts/shard_to_parquet.py
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/1.csv")              # input CSV (after header renaming if you have it)
OUT_DIR = Path("data/parquet_shards")     # output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 5_000  # rows per shard, tune to your RAM/CPU

reader = pd.read_csv(RAW, chunksize=CHUNK)
for i, df in enumerate(reader):
    # if you want to downsample here, do it BEFORE writing parquet
    df.to_parquet(OUT_DIR / f"shard_{i:04d}.parquet", index=False)
    print("wrote", OUT_DIR / f"shard_{i:04d}.parquet")
