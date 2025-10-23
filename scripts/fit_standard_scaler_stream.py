# scripts/fit_standard_scaler_stream.py
"""
Fit a StandardScaler on TRAIN shards only, using incremental .partial_fit
(official scikit-learn pattern for large datasets).

- Chronological split by shard (70% train, 15% val, 15% test) with a 1-shard purge.
- Robust fallback when the shard count is small, so you never end up with 0 train shards.
- Validates that all required feature columns exist in each shard before fitting.
"""

from pathlib import Path
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# --- make 'src' importable when running as a loose script ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import cfg  # uses your feature_cols


def choose_splits(shards):
    """Chronological split with purge, plus a safe fallback if shards are few."""
    n = len(shards)
    purge = 1

    if n >= 6:
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        train_shards = shards[:train_end]
        # val_shards = shards[train_end + purge : val_end]
        # test_shards = shards[val_end + purge :]
    else:
        # Fallback for few shards: first n-2 train, one val, one test
        # (we only need train here, but we mirror the logic for consistency)
        train_shards = shards[:-2] if n > 2 else shards[:1]
        purge = 0

    return train_shards, purge


def main():
    shard_dir = Path("data/parquet_shards")
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(
            f"No shards found in {shard_dir}. Run:\n"
            "  python scripts/shard_to_parquet.py"
        )

    train_shards, purge = choose_splits(shards)
    if not train_shards:
        raise SystemExit("No TRAIN shards selected; reduce CHUNK size or add more data.")

    print(f"[scaler] shards total={len(shards)}  train={len(train_shards)} (purge={purge})")
    print(f"[scaler] features: {cfg.feature_cols}")

    scaler = StandardScaler()

    # Stream over TRAIN shards only
    seen = 0
    for sp in train_shards:
        # load only the feature columns for efficiency
        try:
            X = pd.read_parquet(sp, columns=cfg.feature_cols)
        except Exception as e:
            raise SystemExit(f"Failed reading {sp} with required columns {cfg.feature_cols}:\n{e}")

        missing = [c for c in cfg.feature_cols if c not in X.columns]
        if missing:
            raise SystemExit(f"Shard {sp} is missing required feature columns: {missing}")

        # ensure numeric dtype
        X = X.astype("float32")

        if len(X) == 0:
            continue

        scaler.partial_fit(X.values)
        seen += len(X)
        print(f"[scaler] fitted on {len(X):7d} rows from {sp.name} (cumulative: {seen})")

    if seen == 0:
        raise SystemExit("Scaler saw 0 rows. Check that feature columns are present and CHUNK is sensible.")

    out_path = Path("data/processed/standard_scaler.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_path.as_posix())
    print(f"[scaler] saved: {out_path}  (samples seen: {seen})")


if __name__ == "__main__":
    main()
