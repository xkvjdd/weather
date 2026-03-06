import pandas as pd
import numpy as np
import os

PROJECT_ROOT = r"C:\Users\Ho On Tam\Desktop\bot\weather"
RUN_DIR = os.path.join(PROJECT_ROOT, "data", "Model", "20260306")

RAW_PATH = os.path.join(RUN_DIR, "01_raw_combined.parquet")
PROC_PATH = os.path.join(RUN_DIR, "02_processed_dataset.parquet")

print("\n==============================")
print("LOADING DATA")
print("==============================")

raw = pd.read_parquet(RAW_PATH)
proc = pd.read_parquet(PROC_PATH)

print("RAW rows:", len(raw))
print("PROCESSED rows:", len(proc))


print("\n==============================")
print("COLUMN CHECK")
print("==============================")

required_cols = [
    "timestamp",
    "city",
    "temp",
    "daily_max_temp",
    "remaining_heat",
]

missing = [c for c in required_cols if c not in proc.columns]

if missing:
    print("❌ Missing columns:", missing)
else:
    print("✅ Required columns present")


print("\n==============================")
print("NULL CHECK")
print("==============================")

nulls = proc.isna().mean().sort_values(ascending=False)
print(nulls.head(10))


print("\n==============================")
print("TARGET DISTRIBUTION")
print("==============================")

print(proc["remaining_heat"].describe())

if proc["remaining_heat"].max() < 1:
    print("❌ remaining_heat looks broken")

if proc["remaining_heat"].mean() < 0.1:
    print("❌ remaining_heat distribution suspicious")


print("\n==============================")
print("CITY DISTRIBUTION")
print("==============================")

city_counts = proc.groupby("city").size().sort_values(ascending=False)
print(city_counts)

if city_counts.max() > city_counts.mean() * 5:
    print("⚠ Possible city imbalance")


print("\n==============================")
print("SOLAR FEATURE CHECK")
print("==============================")

solar_cols = [c for c in proc.columns if "solar" in c]

if solar_cols:
    print(proc[solar_cols].describe())
else:
    print("⚠ No solar features detected")


print("\n==============================")
print("TEMP CONSISTENCY CHECK")
print("==============================")

bad = proc[proc["temp"] > proc["daily_max_temp"]]

print("Rows where temp > daily_max:", len(bad))

if len(bad) > 0:
    print("⚠ Daily max logic issue")


print("\n==============================")
print("RANDOM SAMPLE")
print("==============================")

print(proc.sample(10)[[
    "city",
    "timestamp",
    "temp",
    "daily_max_temp",
    "remaining_heat"
]])


print("\n==============================")
print("DATA QUALITY SCORE")
print("==============================")

score = 100

if missing:
    score -= 40

if proc["remaining_heat"].max() < 1:
    score -= 20

if len(bad) > 0:
    score -= 20

if proc["remaining_heat"].isna().mean() > 0.01:
    score -= 20

print("DATA QUALITY SCORE:", score, "/100")

if score >= 90:
    print("✅ Dataset looks GOOD")
elif score >= 70:
    print("⚠ Dataset usable but review issues")
else:
    print("❌ Dataset likely broken")


print("\nCHECK COMPLETE\n")