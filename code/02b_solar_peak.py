import os
import numpy as np
import pandas as pd

DATA = r"C:\Users\Ho On Tam\Desktop\bot\weather\data\Model\20260306\02_processed_dataset_local.parquet"
OUT_DIR = r"C:\Users\Ho On Tam\Desktop\bot\weather\data\Model\20260306"

OUT_DAILY = os.path.join(OUT_DIR, "peak_timing_daily.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "peak_offset_by_city_month.csv")

CITY_TO_LON = {
    "ATL": -84.4277,
    "NYC": -73.8740,
    "CHI": -87.9073,
    "DAL": -96.8518,
    "SEA": -122.3088,
    "MIA": -80.2870,
    "TOR": -79.6306,
    "PAR": 2.5479,
    "SEL": 126.4407,
    "ANK": 32.9951,
    "BUE": -58.5358,
    "LON": -0.4543,
    "WLG": 174.8050,
}

PEAK_HOUR_MIN = 8
PEAK_HOUR_MAX = 20

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading parquet...")
df = pd.read_parquet(DATA)
print(f"Rows loaded: {len(df):,}")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df[df["timestamp"].notna()].copy()

required = ["city", "temp", "timestamp"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if "longitude" not in df.columns or df["longitude"].isna().all():
    df["longitude"] = df["city"].map(CITY_TO_LON)
else:
    df["longitude"] = df["longitude"].fillna(df["city"].map(CITY_TO_LON))

if df["longitude"].isna().any():
    bad = sorted(df.loc[df["longitude"].isna(), "city"].dropna().unique().tolist())
    raise ValueError(f"Missing longitude for cities: {bad}")

# These should already be local-time fields from 02_data_processing.py
df["date"] = df["timestamp"].dt.floor("D")
df["month"] = df["timestamp"].dt.month
df["doy"] = df["timestamp"].dt.dayofyear
df["hour"] = df["timestamp"].dt.hour

if "utc_offset_hours" not in df.columns or df["utc_offset_hours"].isna().any():
    raise ValueError(
        "utc_offset_hours is missing or incomplete. Re-run 02_data_processing.py first "
        "so timestamps are converted to local time properly."
    )

print("Building solar noon if needed...")

if "solar_noon_hour_local" not in df.columns:
    gamma = 2.0 * np.pi / 365.0 * (df["doy"].astype(float) - 1.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    time_offset_minutes = eqtime + 4.0 * df["longitude"] - 60.0 * df["utc_offset_hours"]
    solar_noon_minutes_local = 720.0 - time_offset_minutes
    df["solar_noon_hour_local"] = solar_noon_minutes_local / 60.0

df["solar_noon_ts"] = df["date"] + pd.to_timedelta(df["solar_noon_hour_local"], unit="h")

print("Restricting peak search to daytime hours...")
daytime = df[(df["hour"] >= PEAK_HOUR_MIN) & (df["hour"] <= PEAK_HOUR_MAX)].copy()

if daytime.empty:
    raise ValueError("No rows remain after daytime filter.")

print("Computing daytime daily max temperature...")
daytime["daily_max_temp"] = daytime.groupby(["city", "date"])["temp"].transform("max")

print("Finding centre of the warmest part of each day (no smoothing, daytime only)...")
max_rows = daytime[daytime["temp"] == daytime["daily_max_temp"]].copy()

if max_rows.empty:
    raise ValueError("No max rows found after daytime peak selection.")

peak_mid = (
    max_rows.groupby(["city", "date"], as_index=False)
    .agg(
        peak_ts_num=("timestamp", lambda s: s.astype("int64").mean()),
        n_peak_obs=("timestamp", "size"),
        daily_max_temp=("daily_max_temp", "first"),
        solar_noon_hour_local=("solar_noon_hour_local", "first"),
        solar_noon_ts=("solar_noon_ts", "first"),
    )
)

peak_mid["timestamp"] = pd.to_datetime(peak_mid["peak_ts_num"])
peak_mid = peak_mid.drop(columns=["peak_ts_num"])

peak_mid["month"] = peak_mid["date"].dt.month

peak_mid["peak_hour_local"] = (
    peak_mid["timestamp"].dt.hour
    + peak_mid["timestamp"].dt.minute / 60.0
    + peak_mid["timestamp"].dt.second / 3600.0
)

peak_mid["peak_offset_hours"] = (
    peak_mid["timestamp"] - peak_mid["solar_noon_ts"]
).dt.total_seconds() / 3600.0

peak_mid["abs_peak_offset_hours"] = peak_mid["peak_offset_hours"].abs()

daily_out = peak_mid[
    [
        "city",
        "date",
        "month",
        "timestamp",
        "solar_noon_ts",
        "n_peak_obs",
        "peak_hour_local",
        "solar_noon_hour_local",
        "peak_offset_hours",
        "abs_peak_offset_hours",
        "daily_max_temp",
    ]
].copy()

print("Building city-month summary...")
summary_city_month = (
    daily_out.groupby(["city", "month"], as_index=False)
    .agg(
        n_days=("peak_offset_hours", "size"),
        mean_peak_offset_hours=("peak_offset_hours", "mean"),
        median_peak_offset_hours=("peak_offset_hours", "median"),
        std_peak_offset_hours=("peak_offset_hours", "std"),
        p10_peak_offset_hours=("peak_offset_hours", lambda s: s.quantile(0.10)),
        p90_peak_offset_hours=("peak_offset_hours", lambda s: s.quantile(0.90)),
        mean_abs_peak_offset_hours=("abs_peak_offset_hours", "mean"),
        median_abs_peak_offset_hours=("abs_peak_offset_hours", "median"),
        mean_peak_hour_local=("peak_hour_local", "mean"),
        median_peak_hour_local=("peak_hour_local", "median"),
        mean_solar_noon_hour_local=("solar_noon_hour_local", "mean"),
        median_solar_noon_hour_local=("solar_noon_hour_local", "median"),
        mean_daily_max_temp=("daily_max_temp", "mean"),
        p90_daily_max_temp=("daily_max_temp", lambda s: s.quantile(0.90)),
        mean_n_peak_obs=("n_peak_obs", "mean"),
    )
    .sort_values(["city", "month"])
    .reset_index(drop=True)
)

daily_out.to_csv(OUT_DAILY, index=False)
summary_city_month.to_csv(OUT_SUMMARY, index=False)

print("\nSaved:")
print(OUT_DAILY)
print(OUT_SUMMARY)

print("\nCity-month summary preview:")
print(summary_city_month.head(30).to_string(index=False))

print("\nOverall peak offset summary:")
print(daily_out["peak_offset_hours"].describe())

print("\nDone.")