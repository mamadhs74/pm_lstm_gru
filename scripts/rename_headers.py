# scripts/rename_headers.py
# One-time renamer: reads data/raw/1.csv, writes data/processed/1_clean.csv
# Maps all problematic headers to safe snake_case (no backslashes/brackets/umlauts).

import pandas as pd
from pathlib import Path

RAW = Path("data/raw/1.csv")
OUT = Path("data/processed/1_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- EXACT mapping based on the headers you printed ----
RAW_TO_CLEAN = {
    # already-clean ones -> clean snake_case
    r"Zeitstempel":                  "timestamp",
    r"X_Achse_[s]":                  "time_s",
    r"BR2_Querbeschl_[g]":           "acc_lat_g",
    r"BR5_Bremsdruck_[bar]":         "brake_bar",
    r"Pitch_Angle_[Grad]":           "pitch_deg",
    r"Roll_Angle_[Grad]":            "roll_deg",
    r"Slip_Angle_[Grad]":            "slip_angle_deg",

    # targets (keep as-is so config stays simple)
    r"PM1_ug_per_m3":                "PM1_ug_per_m3",
    r"PM2_5_ug_per_m3":              "PM2_5_ug_per_m3",
    r"PM10_ug_per_m3":               "PM10_ug_per_m3",

    # counts & PN
    r"PM1_count":                    "pm1_count",
    r"PM2_5_count":                  "pm2_5_count",
    r"PM10_count":                   "pm10_count",
    r"PN_PM1 [#/cm³]":               "pn_pm1_per_cm3",
    r"PN_PM2.5 [#/cm³]":             "pn_pm2_5_per_cm3",
    r"PN_PM10 [#/cm³]":              "pn_pm10_per_cm3",

    r"DSN":                          "dsn",
    r"KO2_Aussen_T_[Grad_Celsius]":  "ambient_temp_c",
    r"MO1_Leergas":                  "mo1_leerlauf",
    r"MO1_Verlustmo_[Prozent]":      "loss_torque_pct",
    r"MO5_Verbrauch_[ul]":           "fuel_consumption_ul",
    r"MO6_Ist_Moment_[Prozent]":     "torque_pct",
    r"MO_Istgang":                   "gear",
    r"MO5_max_Moment_[Nm]":          "max_torque_nm",
    r"Distance_[m]":                 "distance_m",
    r"Time_since_midnight_UTC":      "time_since_midnight_utc",
    r"True_Heading_[Grad]":          "true_heading_deg",
    r"Altitude_WGS_84_[m]":          "altitude_m",
    r"Heading":                      "heading",
    r"Latitude":                     "latitude",
    r"Longitude":                    "longitude",

    # THE ODD ONES — use raw strings with a SINGLE backslash
    r"BR8_Laengsbeschl_[m\s2]":      "acc_long_ms2",
    r"KO1_[km\h]":                   "speed_kmh",
    r"BR5_Giergeschw_[Grad\s]":      "yaw_deg_s",

    r"BR3_Rad_VL_[km\h]":            "wheel_fl_kmh",
    r"BR3_Rad_VR_[km\h]":            "wheel_fr_kmh",
    r"BR3_Rad_HL_[km\h]":            "wheel_rl_kmh",
    r"BR3_Rad_HR_[km\h]":            "wheel_rr_kmh",

    r"LW1_Lenk_Gesch_[Grad\s]":      "steer_rate_deg_s",
    r"MO1_Drehzahl_[1\min]":         "rpm",
    r"Velocity_[km\h]":              "velocity_kmh",
}


if not RAW.exists():
    raise FileNotFoundError(f"Missing: {RAW}. Put the raw file there.")

df = pd.read_csv(RAW)
print("\nOriginal columns:")
print(list(df.columns))

df = df.rename(columns=RAW_TO_CLEAN)

# sanity check: warn if anything unmapped still has brackets/backslashes/odd chars
weird = [c for c in df.columns if any(ch in c for ch in "\\[]()°³ ")]
if weird:
    print("\n[WARN] These columns still look odd (consider mapping them too):")
    print(weird)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"\nSaved: {OUT}")
print("Processed columns:")
print(list(pd.read_csv(OUT, nrows=1).columns))
