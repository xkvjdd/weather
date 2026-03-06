import os
import numpy as np
import pandas as pd

DATA = r"C:\Users\Ho On Tam\Desktop\bot\weather\data\Model\20260306\02_processed_dataset.parquet"
OUT_DIR = r"C:\Users\Ho On Tam\Desktop\bot\weather\data\Model\20260306"
OUT_FILE = os.path.join(OUT_DIR, "02_processed_dataset_local.parquet")

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

CITY_TO_TZ = {
    "ATL": "America/New_York",
    "NYC": "America/New_York",
    "CHI": "America/Chicago",
    "DAL": "America/Chicago",
    "SEA": "America/Los_Angeles",
    "MIA": "America/New_York",
    "TOR": "America/Toronto",
    "PAR": "Europe/Paris",
    "SEL": "Asia/Seoul",
    "ANK": "Europe/Istanbul",
    "BUE": "America/Argentina/Buenos_Aires",
    "LON": "Europe/London",
    "WLG": "Pacific/Auckland",
}

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading parquet...")
df = pd.read_parquet(DATA)
print(f"Rows loaded: {len(df):,}")

if "timestamp" not in df.columns:
    raise ValueError("Missing required column: timestamp")
if "city" not in df.columns:
    raise ValueError("Missing required column: city")
if "temp" not in df.columns:
    raise ValueError("Missing required column: temp")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df[df["timestamp"].notna()].copy()

if df.empty:
    raise ValueError("No valid rows after parsing timestamp.")

if "longitude" not in df.columns or df["longitude"].isna().all():
    df["longitude"] = df["city"].map(CITY_TO_LON)
else:
    df["longitude"] = df["longitude"].fillna(df["city"].map(CITY_TO_LON))

if df["longitude"].isna().any():
    bad = sorted(df.loc[df["longitude"].isna(), "city"].dropna().unique().tolist())
    raise ValueError(f"Missing longitude for cities: {bad}")

if df["city"].map(CITY_TO_TZ).isna().any():
    bad = sorted(df.loc[df["city"].map(CITY_TO_TZ).isna(), "city"].dropna().unique().tolist())
    raise ValueError(f"Missing timezone for cities: {bad}")

print("Converting UTC timestamps to local city time...")

city_frames = []

for city, sub in df.groupby("city", sort=False):
    print(f"  {city}: {len(sub):,} rows")
    sub = sub.copy()
    tz_name = CITY_TO_TZ[city]

    ts_utc_aware = pd.to_datetime(sub["timestamp"], errors="coerce", utc=True)
    ts_local_aware = ts_utc_aware.dt.tz_convert(tz_name)

    sub["timestamp_utc"] = ts_utc_aware.dt.tz_localize(None)
    sub["timestamp"] = ts_local_aware.dt.tz_localize(None)

    sub["utc_offset_hours"] = (
        (sub["timestamp"] - sub["timestamp_utc"]).dt.total_seconds() / 3600.0
    )

    city_frames.append(sub)

df = pd.concat(city_frames, ignore_index=True)

if df["utc_offset_hours"].isna().any():
    bad = sorted(df.loc[df["utc_offset_hours"].isna(), "city"].dropna().unique().tolist())
    raise ValueError(f"Missing utc_offset_hours after conversion for cities: {bad}")

print("Building local calendar fields...")

df["date"] = df["timestamp"].dt.floor("D")
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["dayofyear"] = df["timestamp"].dt.dayofyear
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["weekofyear"] = df["timestamp"].dt.isocalendar().week.astype(int)
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

print("Building solar noon...")

gamma = 2.0 * np.pi / 365.0 * (df["dayofyear"].astype(float) - 1.0)
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
df["hour_rel_to_solar_noon"] = (
    (df["timestamp"] - df["solar_noon_ts"]).dt.total_seconds() / 3600.0
)

print("Adding retained features...")

sort_cols = [c for c in ["city", "timestamp"] if c in df.columns]
df = df.sort_values(sort_cols).reset_index(drop=True)

temp_num = pd.to_numeric(df["temp"], errors="coerce")
df["temp_trend_1h"] = temp_num - pd.to_numeric(
    df.groupby("city")["temp"].shift(1), errors="coerce"
)

print("Adding solar heating efficiency feature...")

solar_elev = pd.to_numeric(df.get("solar_elevation_deg"), errors="coerce")
df["solar_heating_efficiency"] = df["temp_trend_1h"] / solar_elev.clip(lower=5)

print("Adding yesterday_max_temp...")

daily_max = (
    df.groupby(["city", "date"], as_index=False)["temp"]
    .max()
    .rename(columns={"temp": "daily_max_temp_from_temp"})
)

daily_max["date"] = pd.to_datetime(daily_max["date"]) + pd.Timedelta(days=1)
daily_max = daily_max.rename(columns={"daily_max_temp_from_temp": "yesterday_max_temp"})

df = df.merge(daily_max, on=["city", "date"], how="left")

print("Dropping removed features if present...")

drop_cols = [
    "temp_trend_2h",
    "temp_trend_3h",
    "cloud_trend_1h",
    "wind_trend_1h",
]
drop_existing = [c for c in drop_cols if c in df.columns]
if drop_existing:
    df = df.drop(columns=drop_existing)

print("Saving...")

df.to_parquet(OUT_FILE, index=False)

print("\nSaved:")
print(OUT_FILE)

print("\nRetained / added columns:")
print([c for c in ["temp_trend_1h", "solar_heating_efficiency", "yesterday_max_temp"] if c in df.columns])

print("\nRemoved columns:")
print(drop_existing)

print("\nColumns preview:")
print(df.columns.tolist())

print("\nData preview:")
print(df.head(10).to_string(index=False))

print("\nDone.")
