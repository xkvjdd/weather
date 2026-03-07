import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Weather Max Dashboard", layout="wide")

DATA_DIR = "data/dashboard"
OBS_PATH = os.path.join(DATA_DIR, "observations.parquet")
MODEL_PATH = os.path.join(DATA_DIR, "forecast_model.parquet")
FORECAST_PATH = os.path.join(DATA_DIR, "forecast_latest.parquet")
RANK_PATH = os.path.join(DATA_DIR, "airport_rankings.parquet")
AUTO_REFRESH_SECONDS = 300

AIRPORT_NAMES = {
    "ATL": "ATL (Hartsfield Jackson Atlanta International Airport)",
    "NYC": "NYC (LaGuardia Airport)",
    "CHI": "CHI (O'Hare International Airport)",
    "DAL": "DAL (Dallas Love Field)",
    "SEA": "SEA (Seattle Tacoma International Airport)",
    "MIA": "MIA (Miami International Airport)",
    "TOR": "TOR (Toronto Pearson International Airport)",
    "PAR": "PAR (Charles de Gaulle Airport)",
    "SEL": "SEL (Incheon International Airport)",
    "ANK": "ANK (Esenboga Airport)",
    "BUE": "BUE (Ezeiza International Airport)",
    "LON": "LON (Heathrow Airport)",
    "WLG": "WLG (Wellington Airport)",
}


def airport_label(code):
    return AIRPORT_NAMES.get(str(code), str(code))


def load_parquet(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def safe_float(x):
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def c_to_f(x):
    x = safe_float(x)
    if x is None:
        return None
    return x * 9 / 5 + 32


def fmt_c(x):
    x = safe_float(x)
    if x is None:
        return "—"
    return f"{x:.1f}°C"


def fmt_f(x):
    x = c_to_f(x)
    if x is None:
        return "—"
    return f"{x:.1f}°F"


def fmt_ts(x):
    if pd.isna(x):
        return "—"
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(x)


def rank_airports(rank_df, obs_df):
    if not rank_df.empty and {"airport", "mae"}.issubset(rank_df.columns):
        tmp = rank_df.copy().sort_values(["mae", "airport"])
        return tmp["airport"].dropna().astype(str).tolist()
    if not obs_df.empty and "airport" in obs_df.columns:
        return sorted(obs_df["airport"].dropna().astype(str).unique().tolist())
    return []


def get_obs(airport, obs_df):
    if obs_df.empty:
        return pd.DataFrame()
    x = obs_df[obs_df["airport"] == airport].copy()
    if x.empty:
        return x
    x["timestamp_local"] = pd.to_datetime(x["timestamp_local"], errors="coerce")
    return x.sort_values("timestamp_local")


def get_today_obs(airport, obs_df):
    x = get_obs(airport, obs_df)
    if x.empty:
        return x
    today = x["timestamp_local"].max().normalize()
    return x[x["timestamp_local"].dt.normalize() == today].copy()


def get_yesterday_obs(airport, obs_df):
    x = get_obs(airport, obs_df)
    if x.empty:
        return x
    today = x["timestamp_local"].max().normalize()
    yesterday = today - pd.Timedelta(days=1)
    x = x[x["timestamp_local"].dt.normalize() == yesterday].copy()
    if x.empty:
        return x
    x["display_time"] = x["timestamp_local"].dt.hour + x["timestamp_local"].dt.minute / 60
    return x


def get_model_hist(airport, model_df):
    if model_df.empty:
        return pd.DataFrame()
    x = model_df[model_df["airport"] == airport].copy()
    if x.empty:
        return x
    x["run_timestamp_local"] = pd.to_datetime(x["run_timestamp_local"], errors="coerce")
    return x.sort_values("run_timestamp_local")


def get_latest_forecast(airport, forecast_df):
    if forecast_df.empty:
        return pd.DataFrame()
    x = forecast_df[forecast_df["airport"] == airport].copy()
    if x.empty:
        return x
    x["pulled_at_local"] = pd.to_datetime(x["pulled_at_local"], errors="coerce")
    return x.sort_values("pulled_at_local").tail(1)


def add_model_absolute_max(model_hist, obs_hist):
    if model_hist.empty:
        return model_hist.copy()

    x = model_hist.copy()

    if "projected_max_temp" not in x.columns:
        x["modelled_max_abs"] = pd.NA
        return x

    x["projected_max_temp"] = pd.to_numeric(x["projected_max_temp"], errors="coerce")

    if obs_hist.empty or "temp" not in obs_hist.columns:
        x["modelled_max_abs"] = x["projected_max_temp"]
        return x

    obs = obs_hist[["timestamp_local", "temp"]].copy()
    obs["timestamp_local"] = pd.to_datetime(obs["timestamp_local"], errors="coerce")
    obs["temp"] = pd.to_numeric(obs["temp"], errors="coerce")
    obs = obs.dropna(subset=["timestamp_local"]).sort_values("timestamp_local")

    mdl = x.sort_values("run_timestamp_local").copy()

    merged = pd.merge_asof(
        mdl,
        obs.rename(columns={"timestamp_local": "obs_timestamp", "temp": "temp_at_run"}),
        left_on="run_timestamp_local",
        right_on="obs_timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2H"),
    )

    merged["temp_at_run"] = pd.to_numeric(merged["temp_at_run"], errors="coerce")
    merged["modelled_max_abs"] = merged["temp_at_run"] + merged["projected_max_temp"]
    merged["modelled_max_abs"] = merged["modelled_max_abs"].where(
        merged["modelled_max_abs"].notna(),
        merged["projected_max_temp"],
    )

    return merged


def make_chart(airport, obs_df, model_df, forecast_df):
    obs_hist = get_obs(airport, obs_df)
    today_obs = get_today_obs(airport, obs_df)
    yesterday_obs = get_yesterday_obs(airport, obs_df)

    model_hist_raw = get_model_hist(airport, model_df)
    model_hist = add_model_absolute_max(model_hist_raw, obs_hist)

    latest_fc = get_latest_forecast(airport, forecast_df)

    fig = go.Figure()

    if not yesterday_obs.empty:
        fig.add_trace(
            go.Scatter(
                x=yesterday_obs["display_time"],
                y=yesterday_obs["temp"],
                mode="lines",
                name="Yesterday",
                line=dict(color="rgba(150,150,150,0.7)", width=2),
            )
        )

    if not today_obs.empty:
        x_today = today_obs["timestamp_local"].dt.hour + today_obs["timestamp_local"].dt.minute / 60
        fig.add_trace(
            go.Scatter(
                x=x_today,
                y=today_obs["temp"],
                mode="lines",
                name="Actual temp",
                line=dict(color="black", width=3),
            )
        )

    if not model_hist.empty:
        for _, row in model_hist.iterrows():
            y = safe_float(row.get("modelled_max_abs"))
            if y is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[0, 24],
                    y=[y, y],
                    mode="lines",
                    name="Modelled max",
                    line=dict(color="#2563eb", width=2),
                    showlegend=False,
                )
            )

    if not latest_fc.empty:
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
        title=dict(
            text=airport_label(airport),
            x=0,
            xanchor="left",
            y=0.97,
            yanchor="top",
            font=dict(size=32),
        ),
        height=340,
        margin=dict(l=10, r=10, t=100, b=10),
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


def airport_stats(airport, obs_df, model_df, forecast_df):
    obs_hist = get_obs(airport, obs_df)
    today_obs = get_today_obs(airport, obs_df)
    model_hist_raw = get_model_hist(airport, model_df)
    model_hist = add_model_absolute_max(model_hist_raw, obs_hist)
    latest_fc = get_latest_forecast(airport, forecast_df)

    current_temp = None
    obs_updated = None

    if not today_obs.empty:
        row = today_obs.iloc[-1]
        current_temp = safe_float(row["temp"])
        obs_updated = row["timestamp_local"]

    model_now = None
    model_updated = None
    if not model_hist.empty and "modelled_max_abs" in model_hist.columns:
        model_now = safe_float(model_hist["modelled_max_abs"].iloc[-1])
        model_updated = model_hist["run_timestamp_local"].iloc[-1]

    fc_avg = fc1 = fc2 = fc3 = None
    fc_updated = None

    if not latest_fc.empty:
        row = latest_fc.iloc[0]
        fc_avg = safe_float(row.get("forecast_avg_max"))
        fc1 = safe_float(row.get("forecast_source_1"))
        fc2 = safe_float(row.get("forecast_source_2"))
        fc3 = safe_float(row.get("forecast_source_3"))
        fc_updated = row.get("pulled_at_local")

    return {
        "current_temp": current_temp,
        "model_now": model_now,
        "fc_avg": fc_avg,
        "fc1": fc1,
        "fc2": fc2,
        "fc3": fc3,
        "obs_updated": obs_updated,
        "model_updated": model_updated,
        "fc_updated": fc_updated,
    }


obs_df = load_parquet(OBS_PATH)
model_df = load_parquet(MODEL_PATH)
forecast_df = load_parquet(FORECAST_PATH)
rank_df = load_parquet(RANK_PATH)

st.title("Airport Max Temperature Dashboard")
st.caption(f"Auto-refresh target: every {AUTO_REFRESH_SECONDS//60} minutes")

airports = rank_airports(rank_df, obs_df)

for airport in airports:
    st.markdown("---")

    chart_col, stat_col = st.columns([4.6, 1.4])

    with chart_col:
        st.plotly_chart(
            make_chart(airport, obs_df, model_df, forecast_df),
            use_container_width=True,
        )

    s = airport_stats(airport, obs_df, model_df, forecast_df)
    airport_obs = get_obs(airport, obs_df)

    with stat_col:
        local_time = "—"
        if not airport_obs.empty:
            local_time = airport_obs["timestamp_local"].max().strftime("%Y-%m-%d %H:%M")

        st.markdown(f"**Local time now**  \n{local_time}")

        st.markdown("")

        st.markdown(
            f"**Current temp**  \n{fmt_c(s['current_temp'])} | {fmt_f(s['current_temp'])}"
        )

        st.markdown("")

        st.markdown(
            f"**Modelled max**  \n{fmt_c(s['model_now'])} | {fmt_f(s['model_now'])}"
        )

        st.markdown("")

        st.markdown(
            f"**Forecast max**  \n"
            f"Avg: {fmt_c(s['fc_avg'])} | {fmt_f(s['fc_avg'])}  \n"
            f"S1: {fmt_c(s['fc1'])} | {fmt_f(s['fc1'])}  \n"
            f"S2: {fmt_c(s['fc2'])} | {fmt_f(s['fc2'])}  \n"
            f"S3: {fmt_c(s['fc3'])} | {fmt_f(s['fc3'])}"
        )

        st.markdown("")

        st.markdown("**Last updated**")

        st.caption(
            f"Obs: {fmt_ts(s['obs_updated'])}\n\n"
            f"Model: {fmt_ts(s['model_updated'])}\n\n"
            f"Forecast: {fmt_ts(s['fc_updated'])}"
        )
