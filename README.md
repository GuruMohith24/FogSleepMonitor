# Sleep Quality Estimation Using Wearable Sensors

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

A Cyber-Physical System (CPS) and Machine Learning project designed to provide real-time, privacy-focused sleep quality estimation. This system integrates hardware sensing with a deep-learning LSTM network to process physiological data locally (Fog Computing), delivering an interpretable sleep score from 0-100.

## 🚀 Project Overview

Traditional sleep monitoring often relies on cloud-based processing, leading to latency issues and privacy concerns. This project addresses these gaps by:

| Feature | Description |
|---|---|
| 🔒 **Local Processing (Fog Node)** | Uses a laptop as a processing hub to ensure data privacy and low-latency feedback |
| ⚡ **Real-Time Sensing** | Captures raw accelerometer and PPG data via hardware microcontrollers |
| 🧠 **Interpretable AI** | Implements a Deep LSTM Network to output sleep quality heuristic scoring and classifications based precisely on sequential physiological factors |

## 🏗️ System Architecture

The project is structured into three distinct layers:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        CYBER/FOG LAYER                              │
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
│           ┌──────────────────────────────────────┐                  │
│           │  Arduino UNO + Sensors               │                  │
│           │  (Raw Physiological Data Acquisition)│                  │
│           └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer Details:**

* **Physical Layer**: PPG Pulse Sensor + Accelerometer + Arduino UNO for raw physiological data acquisition
* **Communication Layer**: USB Serial bridge using PySerial for high-speed data transmission
* **Cyber/Fog Layer**:
    * **Preprocessing**: Feature extraction (Movement Magnitude/Variance, Avg HR, HRV) mapped to rolling sequences
    * **Inference**: A Keras LSTM Neural Network compiling 30-step windows into single sleep scoring
    * **Visualization**: A responsive Streamlit-based web dashboard displaying Good/Poor Sleep classifications 

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Hardware | Arduino UNO, PPG Pulse Sensor, MPU6050 Accelerometer |
| Languages | Python 3.9+, C++ (Arduino Sketch) |
| ML Framework | TensorFlow / Keras (LSTM) |
| Visualization | Streamlit |
| Key Libraries | pandas, numpy, scikit-learn, joblib, pyserial |

## 📂 Project Structure

```text
FogSleepMonitor/
├── 📁 data/                    # Training datasets synthesized
│   └── sleep_sensor_dataset.csv
├── 📁 dashboard/               # Streamlit application UI
│   ├── app.py
│   └── live_data.csv           # Serial data output
├── 📁 fog_node/                # Edge bridging and inference
│   └── fog_service.py          # Data ingestion and prediction
├── 📁 hardware/                # Arduino source code
├── 📁 models/                  # Saved Machine Learning models (Keras/h5)
│   ├── sleep_lstm_model.h5
│   └── scaler.pkl
├── 📄 config.py               # Constants and configuration
├── 📄 train_model.py          # Trains the LSTM and dumps artifacts
├── 📄 predict_realtime.py     # Independent diagnostic checker
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md               # Project documentation
```

## ⚙️ Installation & Setup

1. **Software Installation**

Clone the repository:
```bash
git clone https://github.com/GuruMohith24/Sleep_Quality_Project.git
cd Sleep_Quality_Project/FogSleepMonitor
```

Set up Virtual Environment:
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate
```

Install Dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Usage

### 1. Train Model (Optional)
If you want to view feature importance logic and retrain the LSTM:
```bash
python train_model.py
```
**Expected Output:**
```text
Loaded data from: data/sleep_sensor_dataset.csv

--- Feature Importance ---
Movement Magnitude         0.65
Movement Variance          0.14
Avg Heart Rate             0.12
Hrv                        0.07

Model saved to: models/sleep_lstm_model.h5
```

### 2. Turn on the Fog Stream Bridge
Initialize the fog node which starts listening to hardware / creating mock data testing buffers:
```bash
python fog_node/fog_service.py
```

### 3. Run Real-Time Dashboard
With the fog background process active, launch the real-time Streamlit interface in a separate terminal:
```bash
streamlit run dashboard/app.py
```

## 🧠 Machine Learning Insights

The model uses a Long Short-Term Memory (LSTM) network to sequence 30-interval windows of raw datasets reliably mitigating physiological noise.

### Model Output Constraints
- Classification: Binary (**Good Sleep** vs **Poor Sleep Quality**)
- Score: Heuristic scale from **0-100%**
- Threshold: Sleep quality score ≥ 70% is consistently locked to "Good"

## 🔬 Why This Approach?

| Challenge | Our Solution |
|---|---|
| **Privacy Concerns** | All processing happens locally on the Fog Node (Your PC) |
| **Latency Issues** | Edge computing eliminates web or cloud API round-trips |
| **Noisy Tracking** | LSTMs provide smoothing buffers to identify exact physiological disturbances |
| **Cost** | Low-cost Arduino + custom open-source Python software stack |

## 🎯 Future Enhancements

- [ ] Add more complex deep-learning layers for exact Sleep Stage tracking (REM, Deep, Light)
- [ ] Incorporate SpO2 sensor for oxygen saturation monitoring
- [ ] Connect Streamlit logic to a direct Mobile App for remote night monitoring
