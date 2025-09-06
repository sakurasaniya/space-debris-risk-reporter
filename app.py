import streamlit as st
import pandas as pd
import urllib.request
import urllib.error
from skyfield.api import load, EarthSatellite
from datetime import datetime, timedelta, timezone
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================== FUNCTIONS =====================
def load_tle_from_lines(lines):
    satellites = []
    for i in range(0, len(lines) - 2, 3):
        try:
            name = lines[i].strip()
            l1, l2 = lines[i+1].strip(), lines[i+2].strip()
            sat = EarthSatellite(l1, l2, name, load.timescale())
            satellites.append(sat)
        except Exception:
            continue
    return satellites

def classify_risk(distance):
    if distance < 10: return "🚨 CRITICAL"
    elif distance < 50: return "⚠️ HIGH"
    elif distance < 200: return "🟡 MEDIUM"
    else: return "✅ LOW"

def compute_conjunctions(iss, debris_objects, duration_hours=2, step_minutes=5):
    ts = load.timescale()
    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=duration_hours)

    times, names, distances, velocities = [], [], [], []

    for t_min in range(0, int((end-start).total_seconds()/60), step_minutes):
        t = start + timedelta(minutes=t_min)
        ts_now = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)

        iss_pos = iss.at(ts_now)
        for debris in debris_objects[:50]:
            debris_pos = debris.at(ts_now)
            rel_vec = iss_pos.position.km - debris_pos.position.km
            rel_vel = iss_pos.velocity.km_per_s - debris_pos.velocity.km_per_s
            times.append(t)
            names.append(debris.name)
            distances.append(np.linalg.norm(rel_vec))
            velocities.append(np.linalg.norm(rel_vel))

    df = pd.DataFrame({
        "Time": times,
        "Object": names,
        "Distance (km)": distances,
        "Velocity (km/s)": velocities
    })
    df["Risk"] = df["Distance (km)"].apply(classify_risk)
    return df

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Space Debris Risk Reporter ✨", page_icon="🚀", layout="wide")

# ===================== STAR BACKGROUND CSS =====================
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background: black;
    }
    .stars {
        position: fixed;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
    }
    .star {
        position: absolute;
        background: white;
        border-radius: 50%;
        width: 2px;
        height: 2px;
        opacity: 0.8;
        animation: twinkle 2s infinite alternate, float linear infinite;
        box-shadow: 0 0 6px #fff, 0 0 10px #6C63FF;
    }
    @keyframes twinkle {
        from { opacity: 0.3; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1.2); }
    }
    @keyframes float {
        from { transform: translateY(0px); }
        to { transform: translateY(100vh); }
    }
    </style>
    <div class="stars">
        """ +
        "".join([f'<div class="star" style="top:{np.random.randint(0,100)}%;left:{np.random.randint(0,100)}%;animation-duration:{np.random.randint(20,60)}s;"></div>' for _ in range(120)]) +
        """
    </div>
    """,
    unsafe_allow_html=True
)

# ===================== HEADER =====================
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🚀 Space Debris Risk Reporter</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>A dreamy real-time ISS debris monitoring dashboard ✨</h3>", unsafe_allow_html=True)

# ===================== LOAD & COMPUTE =====================
@st.cache_data(show_spinner="Loading data and computing conjunctions...")
def load_and_compute():
    debris_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    try:
        with urllib.request.urlopen(debris_url) as response:
            debris_lines = response.read().decode('utf-8').splitlines()
        debris_objects = load_tle_from_lines(debris_lines)
    except urllib.error.URLError as e:
        st.error(f"❌ Error loading debris data: {e.reason}")
        return None, None

    iss_url = "https://celestrak.org/NORAD/elements/stations.txt"
    try:
        satellites = load.tle_file(iss_url)
        iss = [sat for sat in satellites if "ISS" in sat.name][0]
    except Exception as e:
        st.error(f"❌ Error loading ISS data: {e}")
        return None, None

    df = compute_conjunctions(iss, debris_objects)
    alerts_df = df.sort_values("Distance (km)").groupby("Object").first().reset_index()
    return df, alerts_df

df, alerts_df = load_and_compute()

# ===================== MAIN DASHBOARD =====================
if df is not None and alerts_df is not None:
    st.success("✅ Data loaded and computations complete!")
    st.write(f"Analyzed `50` debris objects and found `{len(alerts_df)}` potential conjunction alerts.")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📊 Interactive Risk Dashboard")
        fig = px.scatter(
            alerts_df,
            x="Distance (km)",
            y="Velocity (km/s)",
            color="Risk",
            hover_data=["Object", "Time"]
        )
        fig.update_layout(
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🚨 Conjunction Alerts Summary")
        st.dataframe(alerts_df.sort_values("Risk", ascending=False), use_container_width=True)

    # === Placeholder for future GenAI report ===
    st.subheader("📡 Mission Control AI Report (Coming Soon ✨)")
    st.info("This section will generate AI-powered mission reports in the future.")
