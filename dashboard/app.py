import streamlit as st
import pandas as pd
import time
import os
import sys

# Ensure configuration can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(
    page_title="Fog-Based Real-Time Sleep Monitor",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("😴 Fog-Based Real-Time Sleep Monitoring System")
st.markdown("""
**Powered by Edge/Fog Computing & LSTM**  
This dashboard provides real-time tracking of sleep behavior. Data is gathered directly 
from hardware sensors and computed locally on a Fog Node, minimizing latency and preserving privacy.
""")

if not os.path.exists(config.OUTPUT_FILE):
    st.warning("⏳ Awaiting data from Fog Node... Start `fog_service.py` first!")
    st.stop()


def get_sleep_score_color(score):
    """Return a color based on sleep score."""
    if score >= 70:
        return "🟢"
    elif score >= 40:
        return "🟡"
    return "🔴"


# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, config.DASHBOARD_REFRESH_RATE)
chart_window = st.sidebar.slider("Chart History (entries)", 50, 500, 100)

# --- Main Layout ---
st.markdown("### 📊 Live Sensory Data")
col1, col2, col3, col4 = st.columns(4)
placeholder_metric1 = col1.empty()
placeholder_metric2 = col2.empty()
placeholder_metric3 = col3.empty()
placeholder_metric4 = col4.empty()

st.markdown("---")
placeholder_chart1 = st.empty()
placeholder_chart2 = st.empty()

st.markdown("---")
placeholder_alerts = st.empty()

# --- Dashboard Update ---
try:
    df = pd.read_csv(config.OUTPUT_FILE)
    if len(df) > 0:
        latest = df.iloc[-1]

        # Extract metrics
        state = str(latest["Predicted_State"])
        pulse = float(latest["Pulse"])
        confidence = float(latest.get("Confidence", 0.0))
        sleep_score = confidence * 100
        intensity = abs(latest['AcX']) + abs(latest['AcY']) + abs(latest['AcZ'])

        # Metric 1: Sleep State
        with placeholder_metric1.container():
            if state == "Poor Sleep":
                st.error(f"**State:** {state}")
            elif state == "Good Sleep":
                st.success(f"**State:** {state}")
            else:
                st.info(f"**State:** {state}")

        # Metric 2: Sleep Score
        with placeholder_metric2.container():
            emoji = get_sleep_score_color(sleep_score)
            st.metric("Sleep Score", f"{emoji} {sleep_score:.1f}%")

        # Metric 3: Heart Rate
        with placeholder_metric3.container():
            if pulse > 90:
                st.warning(f"❤️ Heart Rate: {pulse:.1f} bpm (Elevated)")
            else:
                st.metric("❤️ Heart Rate", f"{pulse:.1f} bpm")

        # Metric 4: Movement
        with placeholder_metric4.container():
            st.metric("🏃 Movement Intensity", f"{intensity:.3f}")

        # --- Charts ---
        plot_df = df.tail(chart_window).copy()
        plot_df['SleepScore'] = plot_df['Confidence'] * 100
        plot_df['Timestamp'] = pd.to_numeric(plot_df['Timestamp'], errors='coerce')
        plot_df = plot_df.dropna(subset=['Timestamp'])
        # Convert raw millisecond timestamps to readable local time (IST = UTC+5:30)
        from datetime import timedelta
        plot_df['Time'] = (pd.to_datetime(plot_df['Timestamp'], unit='ms') + timedelta(hours=5, minutes=30)).dt.strftime('%H:%M:%S')
        # Average multiple readings per second to get a smooth line (10Hz → 1 point/sec)
        plot_df = plot_df.groupby('Time', sort=False).agg({'Pulse': 'mean', 'SleepScore': 'mean'}).reset_index()

        with placeholder_chart1.container():
            st.subheader("❤️ Heart Rate Over Time")
            st.line_chart(plot_df, x='Time', y='Pulse')

        with placeholder_chart2.container():
            st.subheader("📈 Sleep Score Trends")
            st.area_chart(plot_df, x='Time', y='SleepScore')

        # --- Disturbance Alerts ---
        disturbances = df[
            (df['Disturbance_Reason'] != 'None') &
            (df['Disturbance_Reason'] != 'Low movement and stable HRV')
        ].tail(5)

        with placeholder_alerts.container():
            st.subheader("🚨 Recent Disturbance Alerts")
            if len(disturbances) > 0:
                for _, row in disturbances.iterrows():
                    st.error(
                        f"**Disturbance detected** — "
                        f"Reason: `{row['Disturbance_Reason']}` "
                        f"(Time: {row['Timestamp']})"
                    )
            else:
                st.success("✅ No disturbances detected recently. Sleep is stable.")

except pd.errors.EmptyDataError:
    st.info("Waiting for data from Fog Node...")
except Exception as e:
    st.error(f"Error reading live data: {e}")

time.sleep(refresh_rate)
st.rerun()
