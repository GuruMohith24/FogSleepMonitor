import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(
    page_title="Fog-Based Real-Time Sleep Monitor",
    layout="wide",
)

st.title("Fog-Based Real-Time Sleep Monitoring System")
st.markdown("### Powered by Edge/Fog Computing & LSTM")

DATA_FILE = "live_data.csv"

if not os.path.exists(DATA_FILE):
    st.warning("Awaiting data from Fog Node... Start `fog_service.py`!")
    st.stop()

# Placeholders for metrics and charts
col1, col2, col3 = st.columns(3)
placeholder_metric1 = col1.empty()
placeholder_metric2 = col2.empty()
placeholder_metric3 = col3.empty()

st.markdown("---")
placeholder_chart1 = st.empty()
placeholder_chart2 = st.empty()

placeholder_alerts = st.empty()

def render_dashboard():
    while True:
        try:
            df = pd.read_csv(DATA_FILE)
            if len(df) > 0:
                latest = df.iloc[-1]
                
                # Update Metrics
                with placeholder_metric1.container():
                    st.metric("Current State", latest["Predicted_State"])
                with placeholder_metric2.container():
                    st.metric("Heart Rate (Pulse)", f"{latest['Pulse']:.1f}")
                with placeholder_metric3.container():
                    # Compute movement intensity
                    intensity = abs(latest['AcX']) + abs(latest['AcY']) + abs(latest['AcZ'])
                    st.metric("Movement Intensity", f"{intensity:.1f}")
                
                # Plot charts (last 100 entries for better visualization)
                plot_df = df.tail(100).copy()
                
                # Generate a heuristic sleep score (100 = Stable Sleep, 0 = Awake, 50 = Restless)
                def get_score(state):
                    if state == "Stable Sleep": return 100
                    if state == "Restless Sleep": return 50
                    if state == "Disturbed Sleep": return 25
                    return 0
                    
                plot_df['SleepScore'] = plot_df['Predicted_State'].apply(get_score)
                plot_df['Timestamp'] = pd.to_numeric(plot_df['Timestamp'], errors='coerce')

                with placeholder_chart1.container():
                    st.subheader("Heart Rate Over Time")
                    st.line_chart(plot_df.set_index('Timestamp')['Pulse'])
                    
                with placeholder_chart2.container():
                    st.subheader("Sleep Score Trends")
                    st.line_chart(plot_df.set_index('Timestamp')['SleepScore'])
                
                # Disturbance alerts
                disturbances = df[df['Disturbance_Reason'] != 'None'].tail(5)
                with placeholder_alerts.container():
                    st.subheader("Recent Disturbance Alerts")
                    if len(disturbances) > 0:
                        for idx, row in disturbances.iterrows():
                            st.error(f"⚠️ Disturbance detected at **{row['Timestamp']}** - Reason: **{row['Disturbance_Reason']}**")
                    else:
                        st.success("No disturbances detected recently.")
                        
        except Exception as e:
            st.error(f"Error reading live data: {e}")
            
        time.sleep(1) # Refresh rate

# Run continuous refresh
render_dashboard()
