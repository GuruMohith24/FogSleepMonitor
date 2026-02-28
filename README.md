# Fog-Based Real-Time Sleep Monitoring System

A Cyber-Physical System (CPS) designed to continuously monitor your sleep behavior in real-time. By bridging Arduino hardware sensors with edge/fog-computed Deep Learning (LSTM) algorithms, the system can perform computationally intensive tasks locally with zero latency, eliminating immediate reliance on the cloud.

## Features

- **Arduino Hardware Interface**: C++ code to read real-time data from an accelerometer (MPU6050) and a heart-rate sensor.
- **Fog Compute Service**: A Python backend (`fog_node`) that pre-processes incoming serial data from the hardware, applies a sliding window technique, extracts physical and physiological features, and feeds them into the trained LSTM model.
- **Explainable Disturbance Detection**: Heuristic reasoning modules that explain the causes of sleep disruption (e.g. "Excessive Physical Movement", "Abnormal Heart-Rate (Stress/Nightmare)").
- **LSTM Deep Learning Architecture**: A predictive Long Short-Term Memory network to classify time-series sequences into semantic sleep states ("Awake", "Stable Sleep", "Restless Sleep", "Disturbed Sleep").
- **Streamlit Real-Time Dashboard**: A responsive data application running at the fog layer. Visualizes the real-time scoring data, physiological variables, and historical alerts in an elegant, interactive interface without cloud dependencies.
- **Cloud/Retraining Pipeline**: The collected data sets can periodically be backed up to the cloud and retrained via `train_lstm.py` on large batches of historical records for further personalization.

## Folder Structure

- `hardware/` – Arduino script `arduino_code.ino` for data acquisition. Flash this to the Arduino Uno.
- `model/` – `train_lstm.py` responsible for dataset generation, synthesis, model modeling, and training. Running this script generates `sleep_lstm_model.h5` and `scaler.pkl` in `model/saved_models`.
- `fog_node/` – `fog_service.py` operates continuously, grabbing live data chunks (either through Serial communication or Mock data fallback) and performing interference. It exports predictions into `live_data.csv`.
- `dashboard/` – Holds `app.py`. A streamlit interface that interprets the `live_data.csv`.

## Dependencies

You need Python 3 installed. Install the requirements with:
```bash
pip install -r requirements.txt
```

## How to Run: Initial Run Sequence

1. **Train Model First:** Navigate to `model/` and run `python train_lstm.py`. This simulates a dataset, trains an overarching LSTM model, and places the resulting files in the corresponding folders.
2. **Start the Fog Backend:** Navigate into `fog_node/` and run `python fog_service.py`. This process listens to the Arduino serial interface (ensure `COM3` or the relevant port is correctly defined inside `fog_service.py`). If no hardware is attached, it will intelligently fallback to generating streaming mock-data so you can still view the frontend.
3. **Start the Real-time Dashboard:** Concurrently, navigate to `dashboard/` and run `streamlit run app.py`. This starts the interactive UI in your browser displaying variables processed by your active Fog process.

## GitHub Integration

To push this codebase up to your repository (`https://github.com/guhya-16/FogSleepMonitor`):
1. Create a repository on your GitHub account using your browser (e.g. named `FogSleepMonitor`).
2. Run these commands locally from the overarching project directory:
```bash
git remote add origin https://github.com/guhya-16/FogSleepMonitor.git
git add .
git commit -m "Initial commit for Fog-Based Real-Time Sleep Monitoring System"
git branch -M main
git push -u origin main
```
