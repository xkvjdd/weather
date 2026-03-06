import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Weather Max Dashboard", layout="wide")

DATA_DIR = r"C:\Users\Ho On Tam\Desktop\bot\weather\data\dashboard"
OBS_PATH = os.path.join(DATA_DIR, "observations.parquet")
MODEL_PATH = os.path.join(DATA_DIR, "model_history.parquet")
FORECAST_PATH = os.path.join(DATA_DIR, "forecast_latest.parquet")
RANK_PATH = os.path.join(DATA_DIR, "airport_rankings.parquet")
AUTO_REFRESH_SECONDS = 300


def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def fmt_num(x, digits=1, suffix="°C"):
    x = safe_float(x)
    if x is None:
        return "—"
    return f"{x:.{digits}f}{suffix}"


def fmt_delta(x, digits=1):
    x = safe_float(x)
    if x is None:
        return "—"
    return f"{x:+.{digits}f}"


def fmt_ts(x):
    if pd.isna(x):
        return "—"
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(x)


def rank_airports(rank_df: pd.DataFrame, obs_df: pd.DataFrame):
    if not rank_df.empty and {"airport", "mae"}.issubset(rank_df.columns):
        tmp = rank_df.copy().sort_values(["mae", "airport"])
        return tmp["airport"].dropna().astype(str).tolist()
    if not obs_df.empty and "airport" in obs_df.columns:
        return sorted(obs_df["airport"].dropna().astype(str).unique().tolist())
    return []


def get_obs(airport: str, obs_df: pd.DataFrame) -> pd.DataFrame:
    if obs_df.empty:
        return pd.DataFrame()
    x = obs_df[obs_df["airport"] == airport].copy()
    if x.empty:
        return x
    x["timestamp_local"] = pd.to_datetime(x["timestamp_local"], errors="coerce")
    return x.sort_values("timestamp_local")


def get_today_obs(airport: str, obs_df: pd.DataFrame) -> pd.DataFrame:
    x = get_obs(airport, obs_df)
    if x.empty:
        return x
    today = x["timestamp_local"].max().normalize()
    return x[x["timestamp_local"].dt.normalize() == today].copy()


def get_yesterday_obs(airport: str, obs_df: pd.DataFrame) -> pd.DataFrame:
    x = get_obs(airport, obs_df)
    if x.empty:
        return x
    today = x["timestamp_local"].max().normalize()
    yesterday = today - pd.Timedelta(days=1)
    x = x[x["timestamp_local"].dt.normalize() == yesterday].copy()
    if x.empty:
        return x
    x["display_time"] = x["timestamp_local"].dt.hour + x["timestamp_local"].dt.minute / 60.0
    return x


def get_model_hist(airport: str, model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty:
        return pd.DataFrame()
    x = model_df[model_df["airport"] == airport].copy()
    if x.empty:
        return x
    x["run_timestamp_local"] = pd.to_datetime(x["run_timestamp_local"], errors="coerce")
    return x.sort_values("run_timestamp_local")


def get_latest_forecast(airport: str, forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame()
    x = forecast_df[forecast_df["airport"] == airport].copy()
    if x.empty:
        return x
    x["pulled_at_local"] = pd.to_datetime(x["pulled_at_local"], errors="coerce")
    return x.sort_values("pulled_at_local").tail(1).copy()


def make_chart(airport: str, obs_df: pd.DataFrame, model_df: pd.DataFrame, forecast_df: pd.DataFrame):
    today_obs = get_today_obs(airport, obs_df)
    yesterday_obs = get_yesterday_obs(airport, obs_df)
    model_hist = get_model_hist(airport, model_df)
    latest_fc = get_latest_forecast(airport, forecast_df)

    fig = go.Figure()

    if not yesterday_obs.empty and "temp" in yesterday_obs.columns:
        fig.add_trace(
            go.Scatter(
                x=yesterday_obs["display_time"],
                y=yesterday_obs["temp"],
                mode="lines",
                name="Yesterday",
                line=dict(color="rgba(150,150,150,0.7)", width=2),
            )
        )

    if not today_obs.empty and "temp" in today_obs.columns:
        x_today = today_obs["timestamp_local"].dt.hour + today_obs["timestamp_local"].dt.minute / 60.0
        fig.add_trace(
            go.Scatter(
                x=x_today,
                y=today_obs["temp"],
                mode="lines",
                name="Actual temp",
                line=dict(color="black", width=3),
            )
        )

    if not model_hist.empty and "projected_max_temp" in model_hist.columns:
        blue_scale = [
            "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6",
            "#2563eb", "#1d4ed8", "#1e40af", "#1e3a8a"
        ]
        n = len(model_hist)
        colors = blue_scale[-n:] if n <= len(blue_scale) else [blue_scale[min(int(i * len(blue_scale) / n), len(blue_scale)-1)] for i in range(n)]
        for i, (_, row) in enumerate(model_hist.iterrows()):
            y = safe_float(row.get("projected_max_temp"))
            if y is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[0, 24],
                    y=[y, y],
                    mode="lines",
                    name="Model",
                    line=dict(color=colors[i], width=2),
                    showlegend=(i == n - 1),
                )
            )

    if not latest_fc.empty and "forecast_avg_max" in latest_fc.columns:
        y = safe_float(latest_fc["forecast_avg_max"].iloc[0])
        if y is not None:
            fig.add_trace(
                go.Scatter(
                    x=[0, 24],
                    y=[y, y],
                    mode="lines",
                    name="Forecast avg max",
                    line=dict(color="red", width=4),
                )
            )

    fig.update_layout(
        title=airport,
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            title="Local hour",
            range=[0, 24],
            tickmode="array",
            tickvals=list(range(0, 25, 3)),
            ticktext=[f"{h:02d}:00" for h in range(0, 25, 3)],
        ),
        yaxis=dict(title="Temperature (°C)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )
    return fig


def airport_stats(airport: str, obs_df: pd.DataFrame, model_df: pd.DataFrame, forecast_df: pd.DataFrame):
    today_obs = get_today_obs(airport, obs_df)
    model_hist = get_model_hist(airport, model_df)
    latest_fc = get_latest_forecast(airport, forecast_df)

    current_temp = None
    obs_updated = None
    if not today_obs.empty:
        row = today_obs.tail(1).iloc[0]
        current_temp = safe_float(row.get("temp"))
        obs_updated = row.get("timestamp_local")

    model_now = model_prev = model_updated = None
    if not model_hist.empty:
        model_now = safe_float(model_hist["projected_max_temp"].iloc[-1])
        model_updated = model_hist["run_timestamp_local"].iloc[-1]
        if len(model_hist) >= 2:
            model_prev = safe_float(model_hist["projected_max_temp"].iloc[-2])

    fc_avg = fc1 = fc2 = fc3 = fc_prev = fc_updated = None
    if not latest_fc.empty:
        row = latest_fc.iloc[0]
        fc_avg = safe_float(row.get("forecast_avg_max"))
        fc1 = safe_float(row.get("forecast_source_1"))
        fc2 = safe_float(row.get("forecast_source_2"))
        fc3 = safe_float(row.get("forecast_source_3"))
        fc_updated = row.get("pulled_at_local")

    if not forecast_df.empty:
        x = forecast_df[forecast_df["airport"] == airport].copy()
        if not x.empty:
            x["pulled_at_local"] = pd.to_datetime(x["pulled_at_local"], errors="coerce")
            x = x.sort_values("pulled_at_local")
            if len(x) >= 2 and "forecast_avg_max" in x.columns:
                fc_prev = safe_float(x["forecast_avg_max"].iloc[-2])

    return {
        "current_temp": current_temp,
        "model_now": model_now,
        "model_delta": None if model_now is None or model_prev is None else model_now - model_prev,
        "fc_avg": fc_avg,
        "fc1": fc1,
        "fc2": fc2,
        "fc3": fc3,
        "fc_delta": None if fc_avg is None or fc_prev is None else fc_avg - fc_prev,
        "obs_updated": obs_updated,
        "model_updated": model_updated,
        "fc_updated": fc_updated,
    }


obs_df = load_parquet(OBS_PATH)
model_df = load_parquet(MODEL_PATH)
forecast_df = load_parquet(FORECAST_PATH)
rank_df = load_parquet(RANK_PATH)

st.title("Airport Max Temperature Dashboard")
st.caption(f"Auto-refresh target: every {AUTO_REFRESH_SECONDS // 60} minutes")

airports = rank_airports(rank_df, obs_df)
if not airports:
    st.warning("No airports found. Populate dashboard parquet files first.")
    st.stop()

for airport in airports:
    st.markdown("---")
    chart_col, stat_col = st.columns([4.5, 1.5], gap="medium")

    with chart_col:
        st.plotly_chart(make_chart(airport, obs_df, model_df, forecast_df), use_container_width=True)

    s = airport_stats(airport, obs_df, model_df, forecast_df)
    with stat_col:
        st.subheader(airport)
        st.markdown(f"**Current temp**  
{fmt_num(s['current_temp'])}")
        st.markdown(
            f"**Modelled max**  
{fmt_num(s['model_now'])}  
"
            f"<span style='color:gray;font-size:12px'>{fmt_delta(s['model_delta'])} vs last model</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Forecast max**  
Avg: {fmt_num(s['fc_avg'])}  
"
            f"S1: {fmt_num(s['fc1'])}  
S2: {fmt_num(s['fc2'])}  
S3: {fmt_num(s['fc3'])}  
"
            f"<span style='color:gray;font-size:12px'>{fmt_delta(s['fc_delta'])} vs last forecast</span>",
            unsafe_allow_html=True,
        )
        st.markdown("**Last updated**")
        st.caption(
            f"Obs: {fmt_ts(s['obs_updated'])}

"
            f"Model: {fmt_ts(s['model_updated'])}

"
            f"Forecast: {fmt_ts(s['fc_updated'])}"
        )
