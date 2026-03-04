# 😴 Sleep Quality Estimation Using Wearable Sensors

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **Cyber-Physical System (CPS)** and **Machine Learning** project designed to provide real-time, privacy-focused sleep quality estimation. This system integrates hardware sensing with a deep-learning LSTM network to process physiological data locally (Fog Computing), delivering an interpretable sleep score from 0-100.

---

## 🚀 Project Overview

Traditional sleep monitoring often relies on cloud-based processing, leading to latency issues and privacy concerns. This project addresses these gaps by:

| Feature | Description |
|---------|-------------|
| 🔒 **Local Processing (Fog Node)** | Uses a laptop as a processing hub to ensure data privacy and low-latency feedback |
| ⚡ **Real-Time Sensing** | Captures raw accelerometer and PPG data via Arduino-based microcontrollers |
| 🧠 **Interpretable AI** | Implements a Deep LSTM Network to output sleep quality scoring and classifications based on sequential physiological factors |

---

## 🏗️ System Architecture

The project is structured into three distinct layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CYBER / FOG LAYER                            │
│  ┌───────────────┐   ┌─────────────┐   ┌─────────────────────────┐ │
│  │ Preprocessing │ → │  ML Model   │ → │  Streamlit Dashboard    │ │
│  │ (Sequencing)  │   │   (LSTM)    │   │  (Real-time Visuals)    │ │
│  └───────────────┘   └─────────────┘   └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↑
                    USB Serial (PySerial)
                              ↑
┌─────────────────────────────────────────────────────────────────────┐
│                      PHYSICAL LAYER                                 │
│        ┌────────────────────────────────────────────┐              │
│        │  Arduino UNO + MPU6050 + PPG Pulse Sensor  │              │
│        │  (Raw Physiological Data Acquisition)      │              │
│        └────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer Details:**

1. **Physical Layer:** PPG Pulse Sensor + MPU6050 Accelerometer + Arduino UNO for raw physiological data acquisition
2. **Communication Layer:** USB Serial bridge using `PySerial` for high-speed data transmission (115200 baud)
3. **Cyber/Fog Layer:**
   - **Preprocessing:** Feature extraction (Movement Magnitude/Variance, Avg HR, HRV) via rolling windows
   - **Inference:** A Keras LSTM Neural Network processing 30-step sliding windows into sleep quality scores
   - **Visualization:** A responsive Streamlit-based web dashboard with real-time Good/Poor Sleep classifications

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Hardware** | Arduino UNO R3, PPG Pulse Sensor, MPU6050 Accelerometer |
| **Languages** | Python 3.9+, C++ (Arduino Sketch) |
| **ML Framework** | TensorFlow / Keras (LSTM) |
| **Visualization** | Streamlit |
| **Key Libraries** | `pandas`, `numpy`, `scikit-learn`, `joblib`, `pyserial`, `matplotlib` |

---

## 📂 Project Structure

```
FogSleepMonitor/
├── 📁 data/                       # Training datasets
│   └── sleep_sensor_dataset.csv   # Sensor data (acc, HR, HRV, sleep score)
├── 📁 dashboard/                  # Streamlit application UI
│   ├── app.py                     # Real-time monitoring dashboard
│   └── live_data.csv              # Live serial data output (auto-generated)
├── 📁 fog_node/                   # Edge bridging and inference
│   └── fog_service.py             # Data ingestion, feature extraction & prediction
├── 📁 hardware/                   # Arduino source code
│   └── arduino_code/
│       └── arduino_code.ino       # MPU6050 + Pulse Sensor firmware
├── 📁 models/                     # Saved ML artifacts
│   ├── sleep_lstm_model.h5        # Trained Keras LSTM model
│   └── scaler.pkl                 # MinMaxScaler for feature normalization
├── 📄 config.py                   # Centralized constants and configuration
├── 📄 train_model.py              # LSTM training pipeline with feature engineering
├── 📄 predict_realtime.py         # Standalone diagnostic prediction script
├── 📄 requirements.txt            # Python dependencies
└── 📄 README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Hardware Connection

Connect the sensors to your Arduino UNO:

| Sensor | Pin | Arduino Pin |
|--------|-----|-------------|
| PPG Pulse Sensor | Signal | A0 |
| PPG Pulse Sensor | VCC | 5V |
| PPG Pulse Sensor | GND | GND |
| MPU6050 | SDA | A4 |
| MPU6050 | SCL | A5 |
| MPU6050 | VCC | 5V |
| MPU6050 | GND | GND |

Upload `hardware/arduino_code/arduino_code.ino` using the Arduino IDE.

### 2. Software Installation

**Clone the repository:**
```bash
git clone https://github.com/GuruMohith24/Sleep_Quality_Project.git
cd Sleep_Quality_Project/FogSleepMonitor
```

**Set up Virtual Environment:**
```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Mac/Linux:
source .venv/bin/activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📈 Usage

### 1. Train Model (Optional)
Re-train the LSTM and view feature importance:
```bash
python train_model.py
```

**Expected Output:**
```
Loading data from: data/sleep_sensor_dataset.csv
Extracting features...
Training model...

--- Feature Importance ---
  Movement Magnitude         0.65
  Movement Variance          0.14
  Avg Heart Rate             0.12
  Hrv                        0.07

Model saved to: models/sleep_lstm_model.h5
```

### 2. Start the Fog Node
Initialize the fog processing bridge (connects to hardware or runs mock stream):
```bash
python fog_node/fog_service.py
```

### 3. Run the Dashboard
In a **separate terminal** (with the fog node running):
```bash
streamlit run dashboard/app.py
```

### 4. Quick Diagnostic Test
Test the model independently without hardware:
```bash
python predict_realtime.py
```

---

## 🧠 Machine Learning Insights

The model uses a **Long Short-Term Memory (LSTM)** network to process 30-interval sliding windows of physiological sensor data, effectively capturing temporal patterns and smoothing noisy signals.

### Feature Engineering Pipeline

| Feature | Source | Description |
|---------|--------|-------------|
| `movement_magnitude` | Accelerometer | √(AcX² + AcY² + AcZ²) — total movement intensity |
| `movement_variance` | Derived | Rolling variance (window=10) of movement magnitude |
| `avg_heart_rate` | PPG Sensor | Rolling average heart rate (window=10) |
| `hrv` | PPG Sensor | Heart Rate Variability — std dev of recent HR readings |
| `movement_frequency` | Derived | Count of significant movements in rolling window |
| `sleep_duration` | Context | Total sleep duration (hours) |

### Model Architecture

```
LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1, linear)
```

### Output
- **Score:** Continuous value from **0-100** (higher = better sleep)
- **Classification:** Binary — **Good Sleep** (≥ 70) vs **Poor Sleep** (< 70)
- **Disturbance Detection:** Heuristic analysis for poor sleep episodes

---

## 🔬 Why This Approach?

| Challenge | Our Solution |
|-----------|--------------|
| **Privacy Concerns** | All processing happens locally on the Fog Node (your PC) |
| **Latency Issues** | Edge computing eliminates cloud API round-trips |
| **Noisy Sensor Data** | LSTM sliding windows provide temporal smoothing |
| **Interpretability** | Feature importance + heuristic disturbance explanations |
| **Cost** | Low-cost Arduino + open-source Python stack |

---

## 🎯 Future Enhancements

- [ ] Multi-class sleep staging (REM, Deep, Light, Awake)
- [ ] SpO2 sensor integration for oxygen saturation monitoring
- [ ] Mobile app for remote night monitoring
- [ ] Cloud sync with end-to-end encryption for historical tracking

---

## 👨‍💻 Author

**Guru Mohith**
- GitHub: [@GuruMohith24](https://github.com/GuruMohith24)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ for better sleep quality monitoring</p>
