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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Fog-Based Real-Time Sleep Monitoring System")
st.markdown("""
**Powered by Edge/Fog Computing & LSTM**
This dashboard provides real-time tracking of sleep behavior. Data is gathered directly from hardware sensors and computed locally on a Fog Node, minimizing latency and preserving privacy. 
""")

if not os.path.exists(config.OUTPUT_FILE):
    st.warning("Awaiting data from Fog Node... Start `fog_service.py`!")
    st.stop()

# Placeholders for metrics and charts
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

def render_dashboard():
    while True:
        try:
            df = pd.read_csv(config.OUTPUT_FILE)
            if len(df) > 0:
                latest = df.iloc[-1]
                
                # Setup basic metrics
                state = latest["Predicted_State"]
                pulse = float(latest["Pulse"])
                # Fallback to confidence rendering if the old format CSV is present
                confidence = float(latest["Confidence"]) if "Confidence" in df.columns else 0.0
                intensity = abs(latest['AcX']) + abs(latest['AcY']) + abs(latest['AcZ'])
                
                # Update Metrics
                with placeholder_metric1.container():
                    # Color formatting based on sleep state
                    if state == "Disturbed Sleep":
                        st.error(f"**State:** {state}")
                    elif state == "Restless Sleep":
                        st.warning(f"**State:** {state}")
                    else:
                        st.success(f"**State:** {state}")

                with placeholder_metric2.container():
                    st.metric("Model Confidence", f"{confidence * 100:.1f}%")

                with placeholder_metric3.container():
                    # High pulse alert highlighting
                    if pulse > 650:
                        st.warning(f"Heart Rate: {pulse:.1f} bpm (Elevated)")
                    else:
                        st.metric("Heart Rate (Pulse)", f"{pulse:.1f} bpm")

                with placeholder_metric4.container():
                    st.metric("Movement Intensity", f"{intensity:.1f}")
                
                # Plot charts (last 100 entries for better visualization)
                plot_df = df.tail(100).copy()
                
                # Generate a heuristic sleep score (100 = Stable Sleep, 0 = Awake, 50 = Restless)
                def get_score(state_val):
                    if state_val == "Stable Sleep": return 100
                    if state_val == "Restless Sleep": return 50
                    if state_val == "Disturbed Sleep": return 25
                    return 0
                    
                plot_df['SleepScore'] = plot_df['Predicted_State'].apply(get_score)
                plot_df['Timestamp'] = pd.to_numeric(plot_df['Timestamp'], errors='coerce')

                with placeholder_chart1.container():
                    st.subheader("Heart Rate Over Time")
                    st.line_chart(plot_df.set_index('Timestamp')['Pulse'])
                    
                with placeholder_chart2.container():
                    st.subheader("Sleep Score Trends")
                    st.area_chart(plot_df.set_index('Timestamp')['SleepScore'])
                
                # Disturbance alerts
                disturbances = df[df['Disturbance_Reason'] != 'None'].tail(5)
                with placeholder_alerts.container():
                    st.subheader("🚨 Recent Disturbance Alerts")
                    if len(disturbances) > 0:
                        for idx, row in disturbances.iterrows():
                            # Include timestamp logically formatted 
                            st.error(f"**Disturbance detected** - Reason: `{row['Disturbance_Reason']}` (Time: {row['Timestamp']})")
                    else:
                        st.success("No disturbances detected recently. Sleep is stable.")
                        
        except Exception as e:
            st.error(f"Error reading live data: {e}")
            
        time.sleep(config.DASHBOARD_REFRESH_RATE) # Refresh rate

# Run continuous refresh
render_dashboard()
