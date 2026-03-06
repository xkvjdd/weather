
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

TODAY_STR = "20260306"
PROJECT_ROOT = r"C:\Users\Ho On Tam\Desktop\bot\weather"
MODEL_ROOT_DIR = os.path.join(PROJECT_ROOT, "data", "Model")
RUN_OUTPUT_DIR = os.path.join(MODEL_ROOT_DIR, TODAY_STR)

PROCESSED_PATH = os.path.join(RUN_OUTPUT_DIR, "02_processed_dataset_local.parquet")
PEAK_OFFSET_PATH = os.path.join(RUN_OUTPUT_DIR, "peak_offset_by_city_month.csv")

FEATURE_LIST_PATH = os.path.join(RUN_OUTPUT_DIR, "03_feature_list.txt")
METRICS_PATH = os.path.join(RUN_OUTPUT_DIR, "03_metrics.csv")
METRICS_BY_CITY_PATH = os.path.join(RUN_OUTPUT_DIR, "03_metrics_by_city.csv")
METRICS_BY_MONTH_PATH = os.path.join(RUN_OUTPUT_DIR, "03_metrics_by_month.csv")
TEST_PREDICTIONS_PATH = os.path.join(RUN_OUTPUT_DIR, "03_test_predictions.parquet")
TRAIN_SAMPLE_PATH = os.path.join(RUN_OUTPUT_DIR, "03_training_sample.parquet")
MODEL_SUMMARY_JSON = os.path.join(RUN_OUTPUT_DIR, "03_model_summary.json")
SPLIT_SUMMARY_PATH = os.path.join(RUN_OUTPUT_DIR, "03_split_summary.json")
PERM_IMPORTANCE_PATH = os.path.join(RUN_OUTPUT_DIR, "03_permutation_importance.csv")
BUCKET_FAILURES_PATH = os.path.join(RUN_OUTPUT_DIR, "03_bucket_failures.csv")

TEST_MONTHS = 12
RANDOM_STATE = 42
HORIZON_BUCKETS = ["h9", "h8", "h7", "h6", "h5", "h4", "h3", "h2", "h1"]


def cast_to_float32(x):
    return x.astype(np.float32)


def load_processed() -> pd.DataFrame:
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_PATH}")
    df = pd.read_parquet(PROCESSED_PATH)
    if "timestamp" not in df.columns:
        raise ValueError("Processed dataset must contain a timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()
    if df.empty:
        raise ValueError("Processed dataset is empty after timestamp parsing.")
    return df


def load_peak_offsets() -> pd.DataFrame:
    if not os.path.exists(PEAK_OFFSET_PATH):
        raise FileNotFoundError(f"Peak offset table not found: {PEAK_OFFSET_PATH}")
    peak = pd.read_csv(PEAK_OFFSET_PATH)
    required = ["city", "month", "mean_peak_offset_hours"]
    missing = [c for c in required if c not in peak.columns]
    if missing:
        raise ValueError(f"Peak offset table missing required columns: {missing}")
    peak["month"] = pd.to_numeric(peak["month"], errors="coerce").astype("Int64")
    peak["mean_peak_offset_hours"] = pd.to_numeric(
        peak["mean_peak_offset_hours"], errors="coerce"
    )
    peak = peak.dropna(subset=["city", "month", "mean_peak_offset_hours"]).copy()
    peak["month"] = peak["month"].astype(int)
    peak = peak[["city", "month", "mean_peak_offset_hours"]].drop_duplicates(
        subset=["city", "month"]
    )
    return peak


def ensure_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "city" not in out.columns:
        out["city"] = "Unknown"
    if "station" not in out.columns:
        out["station"] = "Unknown"
    if "country" not in out.columns:
        out["country"] = "Unknown"
    if "region_flag" not in out.columns:
        out["region_flag"] = "Unknown"
    if "is_us_airport" not in out.columns:
        out["is_us_airport"] = 0
    if "is_coastal_airport" not in out.columns:
        out["is_coastal_airport"] = 0

    if "month" not in out.columns:
        out["month"] = out["timestamp"].dt.month
    if "doy" not in out.columns:
        out["doy"] = out["timestamp"].dt.dayofyear

    if "year_num" not in out.columns:
        out["year_num"] = out["timestamp"].dt.year
    if "month_num" not in out.columns:
        out["month_num"] = out["timestamp"].dt.month
    if "day_num" not in out.columns:
        out["day_num"] = out["timestamp"].dt.day

    out["hour_decimal"] = (
        out["timestamp"].dt.hour
        + out["timestamp"].dt.minute / 60.0
        + out["timestamp"].dt.second / 3600.0
    )

    return out


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    doy = pd.to_numeric(out["doy"], errors="coerce")
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    return out


def add_temp_dew_spread(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "temp" in out.columns and "dwpt" in out.columns:
        out["temp_dew_spread"] = pd.to_numeric(out["temp"], errors="coerce") - pd.to_numeric(
            out["dwpt"], errors="coerce"
        )
    return out


def add_lag_delta_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["city", "timestamp"]).copy()
    temp = pd.to_numeric(out["temp"], errors="coerce")

    for h in range(1, 25):
        out[f"temp_lag_{h}h"] = out.groupby("city")["temp"].shift(h)
        out[f"delta_temp_{h}h"] = temp - pd.to_numeric(out[f"temp_lag_{h}h"], errors="coerce")

    if "cldc" in out.columns:
        cldc_num = pd.to_numeric(out["cldc"], errors="coerce")
        for h in [1, 3, 6]:
            out[f"delta_cldc_{h}h"] = cldc_num - pd.to_numeric(
                out.groupby("city")["cldc"].shift(h), errors="coerce"
            )

    if "pres" in out.columns:
        pres_num = pd.to_numeric(out["pres"], errors="coerce")
        for h in [1, 3, 6]:
            out[f"delta_pres_{h}h"] = pres_num - pd.to_numeric(
                out.groupby("city")["pres"].shift(h), errors="coerce"
            )

    for w in [3, 6, 12, 24]:
        roll = out.groupby("city")["temp"].rolling(window=w, min_periods=1)
        out[f"temp_roll_mean_{w}h"] = roll.mean().reset_index(level=0, drop=True)
        out[f"temp_roll_std_{w}h"] = roll.std().reset_index(level=0, drop=True)

    return out


def add_precip_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["city", "timestamp"]).copy()

    if "prcp" not in out.columns:
        out["prcp"] = np.nan

    prcp_num = pd.to_numeric(out["prcp"], errors="coerce").fillna(0.0)
    out["prcp"] = prcp_num
    out["is_raining"] = (prcp_num > 0).astype(int)

    for w in [3, 6, 12, 24]:
        out[f"rain_{w}h_sum"] = (
            out.groupby("city")["prcp"]
            .rolling(window=w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    rain_flag = (out["prcp"] > 0).astype(int)
    hours_since_rain = np.full(len(out), np.nan)

    for city, idx in out.groupby("city").groups.items():
        pos = np.array(list(idx), dtype=int)
        vals = rain_flag.loc[pos].to_numpy()
        arr = np.empty(len(vals), dtype=float)
        last_rain = -1_000_000
        for i, v in enumerate(vals):
            if v == 1:
                last_rain = i
                arr[i] = 0.0
            else:
                arr[i] = np.nan if last_rain < 0 else float(i - last_rain)
        hours_since_rain[pos] = arr

    out["hours_since_rain"] = hours_since_rain
    return out


def add_peak_timing_features(df: pd.DataFrame, peak_offsets: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    out = out.merge(
        peak_offsets,
        how="left",
        left_on=["city", "month"],
        right_on=["city", "month"],
    )

    if out["mean_peak_offset_hours"].isna().any():
        missing = out.loc[out["mean_peak_offset_hours"].isna(), ["city", "month"]].drop_duplicates()
        raise ValueError(
            "Missing mean_peak_offset_hours for some city-month pairs:\n"
            + missing.to_string(index=False)
        )

    if "solar_noon_hour_local" not in out.columns:
        raise ValueError("Processed dataset must contain solar_noon_hour_local.")

    out["solar_peak_hour_local"] = (
        pd.to_numeric(out["solar_noon_hour_local"], errors="coerce")
        + pd.to_numeric(out["mean_peak_offset_hours"], errors="coerce")
    )

    out["hours_from_solar_peak"] = out["hour_decimal"] - out["solar_peak_hour_local"]
    return out


def assign_horizon_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hrs = pd.to_numeric(out["hours_from_solar_peak"], errors="coerce")

    out["horizon_bucket"] = np.select(
        [
            (hrs >= -9) & (hrs < -8),
            (hrs >= -8) & (hrs < -7),
            (hrs >= -7) & (hrs < -6),
            (hrs >= -6) & (hrs < -5),
            (hrs >= -5) & (hrs < -4),
            (hrs >= -4) & (hrs < -3),
            (hrs >= -3) & (hrs < -2),
            (hrs >= -2) & (hrs < -1),
            (hrs >= -1) & (hrs < 0),
        ],
        ["h9", "h8", "h7", "h6", "h5", "h4", "h3", "h2", "h1"],
        default=None,
    )

    out = out[out["horizon_bucket"].notna()].copy()
    return out


def get_feature_columns(df: pd.DataFrame):
    categorical_candidates = ["city", "is_us_airport", "is_coastal_airport"]

    numeric_candidates = [
        "latitude", "longitude", "doy_sin", "doy_cos", "hours_from_solar_peak",
        "temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres",
        "cldc", "coco", "temp_dew_spread", "solar_elevation_deg",
        "solar_hour_angle_deg", "daylight_minutes", "minutes_from_sunrise",
        "minutes_to_sunset", "is_daylight", "is_raining", "rain_3h_sum",
        "rain_6h_sum", "rain_12h_sum", "rain_24h_sum", "hours_since_rain",
        "climo_avg_max_doy", "climo_std_max_doy", "temp_vs_climo_max",
        "temp_trend_1h", "solar_heating_efficiency", "yesterday_max_temp",
    ]

    numeric_candidates += [f"temp_lag_{h}h" for h in range(1, 25)]
    numeric_candidates += [f"delta_temp_{h}h" for h in range(1, 25)]
    numeric_candidates += [f"delta_cldc_{h}h" for h in [1, 3, 6]]
    numeric_candidates += [f"delta_pres_{h}h" for h in [1, 3, 6]]
    numeric_candidates += [f"temp_roll_mean_{w}h" for w in [3, 6, 12, 24]]
    numeric_candidates += [f"temp_roll_std_{w}h" for w in [3, 6, 12, 24]]

    categorical_features = [c for c in categorical_candidates if c in df.columns]
    numeric_features = [c for c in numeric_candidates if c in df.columns and df[c].notna().any()]
    feature_cols = categorical_features + numeric_features
    return feature_cols, categorical_features, numeric_features


def derive_test_window(df: pd.DataFrame, test_months: int = TEST_MONTHS):
    max_ts = df["timestamp"].max()
    min_ts = df["timestamp"].min()
    test_start = (max_ts - pd.DateOffset(months=test_months)).normalize()
    train_end = test_start - pd.Timedelta(seconds=1)
    test_end = max_ts

    if test_start <= min_ts:
        raise ValueError("Not enough history to create a last-12-month test window.")

    return train_end, test_start, test_end


def split_train_test(df: pd.DataFrame):
    train_end, test_start, test_end = derive_test_window(df, TEST_MONTHS)
    train_df = df[df["timestamp"] <= train_end].copy()
    test_df = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty after filtering.")
    if test_df.empty:
        raise ValueError("Test dataset is empty after filtering.")

    return train_df, test_df, train_end, test_start, test_end


def build_pipeline(categorical_features, numeric_features, loss="squared_error", quantile=None):
    cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)),
            ("astype32", FunctionTransformer(cast_to_float32, validate=False)),
        ]
    )
    num = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("astype32", FunctionTransformer(cast_to_float32, validate=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat, categorical_features),
            ("num", num, numeric_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    kwargs = dict(
        max_depth=10,
        max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    categorical_idx = list(range(len(categorical_features))) if categorical_features else None

    if loss == "quantile":
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            categorical_features=categorical_idx,
            **kwargs,
        )
    else:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            categorical_features=categorical_idx,
            **kwargs,
        )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def calc_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def metrics_by_group(df: pd.DataFrame, actual_col: str, pred_col: str, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        if len(g) < 20:
            continue
        rows.append(
            {
                group_col: key,
                "n": int(len(g)),
                "rmse": float(np.sqrt(mean_squared_error(g[actual_col], g[pred_col]))),
                "mae": float(mean_absolute_error(g[actual_col], g[pred_col])),
                "bias": float(np.mean(g[pred_col] - g[actual_col])),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[group_col, "n", "rmse", "mae", "bias"])
    return pd.DataFrame(rows).sort_values(group_col)


def save_feature_list(feature_cols):
    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")


def save_bucket_feature_list(bucket_name: str, feature_cols):
    path = os.path.join(RUN_OUTPUT_DIR, f"03_feature_list_{bucket_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")


def model_path(bucket_name: str, label: str) -> str:
    return os.path.join(RUN_OUTPUT_DIR, f"03_model_{bucket_name}_{label}.joblib")


def run_bucket_models(bucket_name, bucket_df, feature_cols, categorical_features, numeric_features):
    print(f"\n=== Training bucket {bucket_name} ===")
    print(f"Rows in bucket before split: {len(bucket_df):,}")

    train_df, test_df, train_end, test_start, test_end = split_train_test(bucket_df)

    X_train = train_df[feature_cols].copy()
    y_train = pd.to_numeric(train_df["remaining_heat"], errors="coerce")
    X_test = test_df[feature_cols].copy()
    y_test = pd.to_numeric(test_df["remaining_heat"], errors="coerce")

    train_mask = y_train.notna() & np.isfinite(y_train)
    test_mask = y_test.notna() & np.isfinite(y_test)

    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]
    X_test = X_test.loc[test_mask]
    y_test = y_test.loc[test_mask]
    train_df = train_df.loc[train_mask].copy()
    test_df = test_df.loc[test_mask].copy()

    if len(X_train) < 500:
        raise ValueError(f"{bucket_name}: not enough train rows after cleaning ({len(X_train)})")
    if len(X_test) < 100:
        raise ValueError(f"{bucket_name}: not enough test rows after cleaning ({len(X_test)})")

    print(f"{bucket_name} train rows: {len(X_train):,}")
    print(f"{bucket_name} test rows:  {len(X_test):,}")

    metrics_rows = []

    mean_pipe = build_pipeline(categorical_features, numeric_features, loss="squared_error")
    print(f"{bucket_name}: fitting mean model...")
    t0 = time.time()
    mean_pipe.fit(X_train, y_train)
    print(f"{bucket_name}: mean model fit complete in {time.time()-t0:,.1f}s")
    pred_mean = mean_pipe.predict(X_test)

    mean_metrics = calc_metrics(y_test.values, pred_mean)
    mean_metrics["model"] = "mean"
    mean_metrics["horizon_bucket"] = bucket_name
    metrics_rows.append(mean_metrics)

    base_cols = [c for c in ["city", "timestamp", "month_num", "temp", "daily_max_temp", "remaining_heat", "hours_from_solar_peak", "horizon_bucket"] if c in test_df.columns]
    pred_frame = test_df[base_cols].copy()
    pred_frame["pred_remaining_heat_mean"] = pred_mean
    pred_frame["pred_daily_max_mean"] = pred_frame["temp"] + pred_frame["pred_remaining_heat_mean"]
    pred_frame["error_mean"] = pred_frame["pred_remaining_heat_mean"] - pred_frame["remaining_heat"]
    pred_frame["abs_error_mean"] = np.abs(pred_frame["error_mean"])

    quantile_specs = [(0.50, "q50"), (0.90, "q90"), (0.95, "q95")]

    for q, label in quantile_specs:
        pipe = build_pipeline(categorical_features, numeric_features, loss="quantile", quantile=q)
        print(f"{bucket_name}: fitting {label} model...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        print(f"{bucket_name}: {label} model fit complete in {time.time()-t0:,.1f}s")
        pred_q = pipe.predict(X_test)

        metrics = calc_metrics(y_test.values, pred_q)
        metrics["model"] = label
        metrics["horizon_bucket"] = bucket_name
        metrics_rows.append(metrics)

        pred_frame[f"pred_remaining_heat_{label}"] = pred_q
        pred_frame[f"pred_daily_max_{label}"] = pred_frame["temp"] + pred_q
        pred_frame[f"error_{label}"] = pred_q - pred_frame["remaining_heat"]

        joblib.dump(pipe, model_path(bucket_name, label))

    if {"pred_remaining_heat_q50", "pred_remaining_heat_q90"}.issubset(pred_frame.columns):
        pred_frame["pred_interval_p50_p90_width"] = pred_frame["pred_remaining_heat_q90"] - pred_frame["pred_remaining_heat_q50"]

    joblib.dump(mean_pipe, model_path(bucket_name, "mean"))

    city_metrics = metrics_by_group(pred_frame, "remaining_heat", "pred_remaining_heat_mean", "city") if "city" in pred_frame.columns else pd.DataFrame()
    if not city_metrics.empty:
        city_metrics["horizon_bucket"] = bucket_name

    month_metrics = metrics_by_group(pred_frame, "remaining_heat", "pred_remaining_heat_mean", "month_num") if "month_num" in pred_frame.columns else pd.DataFrame()
    if not month_metrics.empty:
        month_metrics["horizon_bucket"] = bucket_name

    pi_n = min(500, len(X_test))
    print(f"{bucket_name}: running permutation importance on {pi_n:,} rows...")
    X_pi = X_test.sample(pi_n, random_state=RANDOM_STATE) if len(X_test) > pi_n else X_test.copy()
    y_pi = y_test.loc[X_pi.index]
    pi = permutation_importance(
        mean_pipe,
        X_pi,
        y_pi,
        n_repeats=1,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )
    print(f"{bucket_name}: permutation importance complete")
    pi_df = pd.DataFrame(
        {
            "feature": X_pi.columns,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
            "horizon_bucket": bucket_name,
        }
    ).sort_values("importance_mean", ascending=False)

    train_sample_cols = [c for c in (feature_cols + ["remaining_heat", "daily_max_temp", "timestamp"]) if c in train_df.columns]
    train_sample = train_df[train_sample_cols].head(50000).copy()
    train_sample["horizon_bucket"] = bucket_name

    split_summary = {
        "horizon_bucket": bucket_name,
        "test_months": TEST_MONTHS,
        "min_timestamp": str(bucket_df["timestamp"].min()),
        "max_timestamp": str(bucket_df["timestamp"].max()),
        "train_end": str(train_end),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "n_bucket_rows": int(len(bucket_df)),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
    }

    return {
        "metrics_df": pd.DataFrame(metrics_rows),
        "pred_frame": pred_frame,
        "city_metrics": city_metrics,
        "month_metrics": month_metrics,
        "pi_df": pi_df,
        "train_sample": train_sample,
        "split_summary": split_summary,
    }


def main():
    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

    df = load_processed()
    peak_offsets = load_peak_offsets()

    df = ensure_basic_columns(df)
    df = add_seasonal_features(df)
    df = add_temp_dew_spread(df)
    df = add_lag_delta_roll_features(df)
    df = add_precip_features(df)
    df = add_peak_timing_features(df, peak_offsets)
    df = assign_horizon_bucket(df)

    print(f"Loaded and engineered dataset -> {len(df):,} rows")

    required = ["timestamp", "remaining_heat", "temp", "daily_max_temp", "hours_from_solar_peak"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Processed dataset missing required columns: {missing_required}")

    feature_cols, categorical_features, numeric_features = get_feature_columns(df)
    if not feature_cols:
        raise ValueError("No usable model features found in processed dataset.")

    trend_cols_found = [c for c in ["temp_trend_1h", "solar_heating_efficiency", "yesterday_max_temp"] if c in feature_cols]
    print(f"Retained special features available: {trend_cols_found}")

    save_feature_list(feature_cols)
    print(f"Saved overall feature list -> {FEATURE_LIST_PATH}")
    print(f"Using {len(feature_cols)} features ({len(categorical_features)} categorical, {len(numeric_features)} numeric)")

    all_metrics, all_preds, all_city_metrics, all_month_metrics, all_pi, all_train_samples, all_split_summaries = [], [], [], [], [], [], []
    bucket_failures = []

    bucket_counts = (
        df["horizon_bucket"].value_counts(dropna=False)
        .reindex(HORIZON_BUCKETS, fill_value=0)
        .rename_axis("horizon_bucket")
        .reset_index(name="n_rows")
    )
    print("Rows by horizon bucket:")
    print(bucket_counts.to_string(index=False))

    for bucket_name in HORIZON_BUCKETS:
        bucket_df = df[df["horizon_bucket"] == bucket_name].copy()

        if bucket_df.empty:
            print(f"Skipping {bucket_name}: no rows")
            bucket_failures.append({"horizon_bucket": bucket_name, "status": "skipped", "reason": "no rows in bucket", "n_bucket_rows": 0})
            continue

        save_bucket_feature_list(bucket_name, feature_cols)

        try:
            result = run_bucket_models(bucket_name, bucket_df, feature_cols, categorical_features, numeric_features)
        except Exception as e:
            print(f"Skipping {bucket_name}: {e}")
            bucket_failures.append({
                "horizon_bucket": bucket_name,
                "status": "failed",
                "reason": str(e),
                "n_bucket_rows": int(len(bucket_df)),
                "min_timestamp": str(bucket_df['timestamp'].min()),
                "max_timestamp": str(bucket_df['timestamp'].max()),
            })
            continue

        all_metrics.append(result["metrics_df"])
        all_preds.append(result["pred_frame"])
        if not result["city_metrics"].empty:
            all_city_metrics.append(result["city_metrics"])
        if not result["month_metrics"].empty:
            all_month_metrics.append(result["month_metrics"])
        all_pi.append(result["pi_df"])
        all_train_samples.append(result["train_sample"])
        all_split_summaries.append(result["split_summary"])

    failures_df = pd.DataFrame(bucket_failures)
    if not failures_df.empty:
        failures_df.to_csv(BUCKET_FAILURES_PATH, index=False)
        print(f"Saved bucket failures -> {BUCKET_FAILURES_PATH}")

    if not all_metrics:
        if not failures_df.empty:
            raise ValueError("No horizon models were trained successfully. See bucket failures: " + BUCKET_FAILURES_PATH)
        raise ValueError("No horizon models were trained successfully.")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)
    city_metrics_df = pd.concat(all_city_metrics, ignore_index=True) if all_city_metrics else pd.DataFrame()
    month_metrics_df = pd.concat(all_month_metrics, ignore_index=True) if all_month_metrics else pd.DataFrame()
    pi_df = pd.concat(all_pi, ignore_index=True)
    train_sample_df = pd.concat(all_train_samples, ignore_index=True)

    metrics_df.to_csv(METRICS_PATH, index=False)
    preds_df.to_parquet(TEST_PREDICTIONS_PATH, index=False)
    city_metrics_df.to_csv(METRICS_BY_CITY_PATH, index=False)
    month_metrics_df.to_csv(METRICS_BY_MONTH_PATH, index=False)
    pi_df.to_csv(PERM_IMPORTANCE_PATH, index=False)
    train_sample_df.to_parquet(TRAIN_SAMPLE_PATH, index=False)

    with open(SPLIT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(all_split_summaries, f, indent=2)

    with open(MODEL_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed_path": PROCESSED_PATH,
                "peak_offset_path": PEAK_OFFSET_PATH,
                "n_total_rows_after_bucket_filter": int(len(df)),
                "n_features": int(len(feature_cols)),
                "categorical_features": categorical_features,
                "numeric_features": numeric_features,
                "feature_cols": feature_cols,
                "trend_features": trend_cols_found,
                "test_months": TEST_MONTHS,
                "horizon_buckets": HORIZON_BUCKETS,
            },
            f,
            indent=2,
        )

    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved metrics by city -> {METRICS_BY_CITY_PATH}")
    print(f"Saved metrics by month -> {METRICS_BY_MONTH_PATH}")
    print(f"Saved test predictions -> {TEST_PREDICTIONS_PATH}")
    print(f"Saved training sample -> {TRAIN_SAMPLE_PATH}")
    print(f"Saved permutation importance -> {PERM_IMPORTANCE_PATH}")
    print(f"Saved split summary -> {SPLIT_SUMMARY_PATH}")
    print(f"Saved model summary -> {MODEL_SUMMARY_JSON}")
    print("03_model_fitting complete.")


if __name__ == "__main__":
    main()
