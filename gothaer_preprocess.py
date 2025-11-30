# gothaer_preprocess.py
import pandas as pd
import numpy as np
import re

CSV_PATH = "GothaerPlatz_017_vehicles.csv"

# Column definitions
VEH = "NO"
LAT = "LATWGS84"
LON = "LONGWGS84"

SPEED = "SPEED_NUM"
ACC = "ACCELERATION_NUM"

def parse_coord(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", ".")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    return float(nums[0]) if nums else np.nan

def load_gothaer(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, sep=";")

    # parse coords
    df[LAT] = df[LAT].apply(parse_coord)
    df[LON] = df[LON].apply(parse_coord)

    df = df.dropna(subset=[LAT, LON])

    # convert to meters
    lat0 = df[LAT].mean()
    lon0 = df[LON].mean()

    df["x_m"] = (df[LON] - lon0) * 111320 * np.cos(np.radians(lat0))
    df["y_m"] = (df[LAT] - lat0) * 110540

    # ensure numeric
    df[SPEED] = pd.to_numeric(df[SPEED], errors="coerce")
    df[ACC] = pd.to_numeric(df[ACC], errors="coerce")

    df = df.sort_values(["NO", "SIMSEC_NUM"])  # sorted by time
    df = df.reset_index(drop=True)
    return df
