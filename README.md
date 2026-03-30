# Fog-Based Real-Time Sleep Quality Monitoring System

A **Cyber-Physical System** that monitors sleep quality in real-time using wearable sensors, **Fog Computing**, and a **TensorFlow Lite LSTM** model — all processed locally with zero cloud dependency.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    FOG LAYER (Laptop)                 │
│                                                      │
│  Preprocessing  →  TFLite LSTM  →  Streamlit Dashboard│
│  (Feature Eng.)    (64→32 units)    (Live Score + HR) │
└──────────────────────────────────────────────────────┘
                        ↑
              USB Serial (115200 baud)
                        ↑
┌──────────────────────────────────────────────────────┐
│                 EDGE LAYER (Arduino)                  │
│                                                      │
│  Arduino UNO + MPU-6050 (Accel) + PPG Pulse Sensor   │
│  Output: timestamp, AcX, AcY, AcZ, Pulse @ 10Hz     │
└──────────────────────────────────────────────────────┘
```

**Flow:** Arduino reads sensors → sends CSV over serial → Laptop (Fog Node) runs TFLite inference → Streamlit shows live results.

---

## Dataset

**MMASH** (PhysioNet) — 22 real subjects, 1.4M data points of accelerometer + heart rate during sleep.

> Citation: Schmidt, P. & Reiss, A. (2018). *MMASH Dataset*. PhysioNet. https://doi.org/10.24432/C57K5T

---

## ML Model

| Detail | Value |
|--------|-------|
| **Architecture** | LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1) |
| **Input** | 30 timesteps × 6 features (movement magnitude, variance, avg HR, HRV, movement frequency, sleep duration) |
| **Output** | Sleep Score 0–100 → Good (≥70) / Poor (<70) |
| **Format** | TensorFlow Lite (`.tflite`) — optimized for edge inference |
| **Scaler** | MinMaxScaler fitted on training set, serialized as `scaler.pkl` |

---

## Project Structure

```
FogSleepMonitor/
├── dashboard/app.py              # Streamlit real-time dashboard
├── fog_node/fog_service.py       # Fog processing + TFLite inference
├── hardware/arduino_code/        # Arduino .ino firmware
├── models/
│   ├── sleep_model.tflite        # Optimized LSTM model
│   ├── sleep_lstm_model.h5       # Original Keras model
│   └── scaler.pkl                # MinMaxScaler
├── config.py                     # All settings
├── prepare_mmash_dataset.py      # Dataset downloader + preprocessor
├── train_model.py                # LSTM training pipeline
├── convert_to_tflite.py          # Keras → TFLite conversion
├── predict_realtime.py           # Standalone test script
└── requirements.txt              # Python dependencies
```

---

## Setup & Run

### 1. Install

```bash
git clone https://github.com/guhya-16/FogSleepMonitor.git
cd FogSleepMonitor
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt
```

### 2. Train (optional — pre-trained model included)

```bash
python prepare_mmash_dataset.py   # Downloads MMASH dataset
python train_model.py             # Trains LSTM
python convert_to_tflite.py       # Converts to TFLite
```

### 3. Hardware Wiring

| MPU-6050 | Arduino | | Pulse Sensor | Arduino | | Actuators | Arduino |
|----------|---------|---|-------------|---------|---|-----------|---------|
| VCC | 5V | | + | 5V | | LED (+) | Pin 13 |
| GND | GND | | – | GND | | Buzzer (+) | Pin 8 |
| SDA | A4 | | S | A0 | | Both (–) | GND |
| SCL | A5 | | | | | | |

Flash `hardware/arduino_code/arduino_code.ino` via Arduino IDE (baud: 115200).

### 4. Run

**Terminal 1** — Fog Node:
```bash
$env:SLEEP_SERIAL_PORT="COM3"    # Set your port
python fog_node/fog_service.py   # Falls back to mock data if no Arduino
```

**Terminal 2** — Dashboard:
```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** to view the live dashboard.

---

## Why Fog Computing?

- **Privacy** — Health data stays on your machine, never uploaded to cloud
- **Low Latency** — No network round-trips, instant predictions
- **Offline** — Works without internet after model is trained
- **Low Cost** — Arduino (~₹500) + free Python stack

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Edge** | Arduino UNO, MPU-6050, PPG Pulse Sensor |
| **Fog** | Python, TensorFlow Lite, NumPy, Pandas, Scikit-learn |
| **Dashboard** | Streamlit |
| **Communication** | PySerial (USB Serial) |

---

## Contributors

- [@GuruMohith24](https://github.com/GuruMohith24)
- [@guhya-16](https://github.com/guhya-16)

## License

MIT
