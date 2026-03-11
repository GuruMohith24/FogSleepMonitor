# 🎬 Demo Instructions (Software-Only Mode)

These instructions are for demonstrating the project **without physical hardware** (Arduino + Sensors).  
The system uses a built-in **mock sensor stream** that generates realistic physiological data.

---

## Prerequisites

- Python 3.9+ installed
- Virtual environment set up with all dependencies

---

## Step 1: Open the Project

Open a terminal (PowerShell / CMD / VS Code Terminal) and navigate to the project:

```powershell
cd C:\Users\HP\OneDrive\Desktop\sleep-project\FogSleepMonitor
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate
```

---

## Step 2: Start the Fog Node (Terminal 1)

Set the serial port to a non-existent port to trigger **mock mode**, then start the fog service:

```powershell
$env:SLEEP_SERIAL_PORT="COM99"
python fog_node/fog_service.py
```

**What you should see:**
```
INFO - Initializing Fog Node processing...
INFO - Model and Scaler loaded successfully.
WARNING - Could not connect to Arduino on COM99. Using mock data stream.
INFO - [timestamp] State: Good Sleep (Score: 74.9%) | Reason: Low movement and stable HRV
INFO - [timestamp] State: Good Sleep (Score: 75.3%) | Reason: Low movement and stable HRV
...
```

> ⚠️ **Do NOT close this terminal.** Keep it running in the background.

---

## Step 3: Start the Dashboard (Terminal 2)

Open a **second terminal**, activate the venv again, and start Streamlit:

```powershell
cd C:\Users\HP\OneDrive\Desktop\sleep-project\FogSleepMonitor
.venv\Scripts\Activate
streamlit run dashboard/app.py
```

**What you should see:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

A browser window will open automatically. If it doesn't, open **http://localhost:8501** manually.

---

## Step 4: Show the Dashboard

The dashboard will display:

- **State:** Good Sleep / Poor Sleep
- **Sleep Score:** 0–100% (with green/red indicator)
- **Heart Rate:** Live BPM readings (~55–70 BPM for simulated calm sleep)
- **Movement Intensity:** Low values (~0.1–0.2)
- **Heart Rate Over Time:** Live line chart
- **Sleep Score Trends:** Live area chart

> The dashboard auto-refreshes every 1 second. Use the sidebar to adjust refresh rate and chart history.

---

## Step 5: Quick Standalone Test (Optional)

To show the model prediction without the dashboard:

```powershell
python predict_realtime.py
```

**Output:**
```
Sample  5/30: buffering...
Sample 30/30: predicting...
--- Final Prediction ---
  Sleep Score   : 68.8
  Classification: Poor Sleep
  Reason        : High movement or unstable HRV
```

---

## Step 6: Stop Everything

- **Terminal 1 (Fog Node):** Press `Ctrl + C`
- **Terminal 2 (Dashboard):** Press `Ctrl + C`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Fog node stuck with no output | You forgot `$env:SLEEP_SERIAL_PORT="COM99"`. Restart with that line first. |
| Dashboard shows empty "Live Sensory Data" | Fog node isn't running. Start Terminal 1 first, then Terminal 2. |
| `ModuleNotFoundError: No module named 'tensorflow'` | Virtual environment not activated. Run `.venv\Scripts\Activate` first. |
| Streamlit asks for email | Just press Enter to skip. |
| Dashboard opens on port 8502 instead of 8501 | Another Streamlit instance is running. Close all terminals and start fresh. |
