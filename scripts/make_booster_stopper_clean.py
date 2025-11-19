# scripts/make_booster_stopper_clean.py
#
# Build booster_clean.csv and stopper_clean.csv from the raw Excel files
# by merging the relevant sheets (Zeiten, T6_CAN, VBox3i, Umwelt)
# and then renaming columns into the same style as 1.csv / 1_clean.csv.

import pandas as pd
from pathlib import Path

# ---- This is your RAW_TO_CLEAN dict from rename_headers.py (copy-pasted) ----
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

    # counts & PN (may be missing for Booster/Stopper – that's OK)
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

def build_from_excel(xlsx_path: Path, out_csv: Path):
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing Excel: {xlsx_path}")

    print(f"\n=== Building {out_csv.name} from {xlsx_path.name} ===")

    # 1) Read the relevant sheets
    df_time = pd.read_excel(xlsx_path, sheet_name="Zeiten")
    df_can  = pd.read_excel(xlsx_path, sheet_name="T6_CAN")
    df_vbox = pd.read_excel(xlsx_path, sheet_name="VBox3i")
    df_umw  = pd.read_excel(xlsx_path, sheet_name="Umwelt")

    # Sanity: all same length?
    print("Shapes:",
          "Zeiten", df_time.shape,
          "T6_CAN", df_can.shape,
          "VBox3i", df_vbox.shape,
          "Umwelt", df_umw.shape)

    # 2) Fix German decimal comma in X_Achse_[s]
    if df_time["X_Achse_[s]"].dtype == object:
        df_time["X_Achse_[s]"] = (
            df_time["X_Achse_[s]"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    # 3) Merge all sheets row-wise (they are already aligned by sample index)
    df_all = pd.concat([df_time, df_can, df_vbox, df_umw], axis=1)

    # 4) Create PM ug/m³ columns from Umwelt mg/m³
    #    PMx_[mg\m3] * 1000 = PMx_ug_per_m3
    if "PM1_[mg\\m3]" in df_all.columns:
        df_all["PM1_ug_per_m3"] = df_all["PM1_[mg\\m3]"] * 1000.0
    if "PM2.5_[mg\\m3]" in df_all.columns:
        df_all["PM2_5_ug_per_m3"] = df_all["PM2.5_[mg\\m3]"] * 1000.0
    if "PM10_[mg\\m3]" in df_all.columns:
        df_all["PM10_ug_per_m3"] = df_all["PM10_[mg\\m3]"] * 1000.0

    # 5) Rename columns to the same canonical names as 1_clean.csv
    df_all = df_all.rename(columns=RAW_TO_CLEAN)

    # Optional: warn if any weird headers remain (for debugging)
    weird = [c for c in df_all.columns if any(ch in c for ch in "\\[]()°³ ")]
    if weird:
        print("[WARN] Still odd-looking columns after rename:")
        print(weird)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False)
    print(f"[OK] Saved: {out_csv}")
    print(f"Columns: {list(df_all.columns)}")


if __name__ == "__main__":
    booster_xlsx = Path("data/raw/2025_09_03_10_06_44_Messfahrt_booster.xlsx")
    stopper_xlsx = Path("data/raw/2025_09_03_10_14_44_Messfahrt_stoppper.xlsx")

    booster_out = Path("data/processed/booster_clean.csv")
    stopper_out = Path("data/processed/stopper_clean.csv")

    build_from_excel(booster_xlsx, booster_out)
    build_from_excel(stopper_xlsx, stopper_out)
