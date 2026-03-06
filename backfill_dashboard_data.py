import os
import joblib
import pandas as pd

PROJECT_ROOT = r"C:\Users\Ho On Tam\Desktop\bot\weather"
MODEL_RUN_DIR = os.path.join(PROJECT_ROOT, "data", "Model", "20260306")
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "data", "dashboard")

PROCESSED_PATH = os.path.join(MODEL_RUN_DIR, "02_processed_dataset_local.parquet")
RANK_PATH = os.path.join(MODEL_RUN_DIR, "03_metrics_by_city.csv")
OUT_MODEL_HISTORY = os.path.join(DASHBOARD_DIR, "model_history.parquet")
OUT_OBS = os.path.join(DASHBOARD_DIR, "observations.parquet")
OUT_RANK = os.path.join(DASHBOARD_DIR, "airport_rankings.parquet")
HORIZON_BUCKETS = ["h9", "h8", "h7", "h6", "h5", "h4", "h3", "h2", "h1"]


def model_path(bucket: str) -> str:
    return os.path.join(MODEL_RUN_DIR, f"03_model_{bucket}_mean.joblib")


def get_feature_list(bucket: str):
    p = os.path.join(MODEL_RUN_DIR, f"03_feature_list_{bucket}.txt")
    if not os.path.exists(p):
        p = os.path.join(MODEL_RUN_DIR, "03_feature_list.txt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Feature list not found for {bucket}")
    with open(p, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def main():
    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    df = pd.read_parquet(PROCESSED_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()

    max_ts = df["timestamp"].max()
    start_ts = max_ts - pd.Timedelta(days=2)
    df = df[df["timestamp"] >= start_ts].copy()

    obs = df[["city", "timestamp", "temp"]].copy()
    obs = obs.rename(columns={"city": "airport", "timestamp": "timestamp_local"})
    obs.to_parquet(OUT_OBS, index=False)

    if os.path.exists(RANK_PATH):
        rank = pd.read_csv(RANK_PATH)
        if {"city", "mae"}.issubset(rank.columns):
            rank = rank[["city", "mae"]].rename(columns={"city": "airport"})
            rank = rank.sort_values(["mae", "airport"]).reset_index(drop=True)
            rank.to_parquet(OUT_RANK, index=False)

    rows = []
    for bucket in HORIZON_BUCKETS:
        mp = model_path(bucket)
        if not os.path.exists(mp):
            continue
        model = joblib.load(mp)
        feature_cols = get_feature_list(bucket)

        sub = df[df["horizon_bucket"] == bucket].copy()
        if sub.empty:
            continue

        sub = sub.sort_values(["city", "timestamp"]).copy()
        sub["hour_floor"] = sub["timestamp"].dt.floor("H")
        idx = sub.groupby(["city", "hour_floor"])["timestamp"].idxmax()
        sub = sub.loc[idx].copy()

        X = sub[[c for c in feature_cols if c in sub.columns]].copy()
        pred = model.predict(X)

        out = pd.DataFrame({
            "airport": sub["city"].values,
            "run_timestamp_local": sub["timestamp"].values,
            "projected_max_temp": pd.to_numeric(sub["temp"], errors="coerce").values + pred,
            "current_temp": pd.to_numeric(sub["temp"], errors="coerce").values,
            "is_backfill": True,
            "horizon_bucket": bucket,
        })
        rows.append(out)

    if rows:
        model_hist = pd.concat(rows, ignore_index=True).sort_values(["airport", "run_timestamp_local"])
        model_hist.to_parquet(OUT_MODEL_HISTORY, index=False)

    print("Saved:")
    print(OUT_OBS)
    print(OUT_MODEL_HISTORY)
    print(OUT_RANK)


if __name__ == "__main__":
    main()
