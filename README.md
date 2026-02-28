# Fog-Based Real-Time Sleep Monitoring System

A Cyber-Physical System (CPS) designed to continuously monitor your sleep behavior in real-time. By bridging Arduino hardware sensors with edge/fog-computed Deep Learning algorithms (LSTM), the system can perform computationally intensive tasks locally with zero latency, minimizing cloud dependency while retaining highly accurate results.

---

## 1. Introduction
Sleep quality is an essential aspect of human health. Modern monitoring tools often require uploading massive, continuous streams of raw physiological data to the cloud. This incurs latency, creates bandwidth bottlenecks, and introduces data privacy risks. This project demonstrates a **Fog Computing** architecture applied to a **Cyber-Physical System (CPS)**. By placing the analytical power—a Long Short-Term Memory (LSTM) deep learning network—near the edge layer, predictions regarding a user's sleep state occur instantaneously and locally. This system combines hardware sensors (accelerometer, heart-rate) with a fog node to execute real-time, explainable sleep disturbance predictions without uploading plaintext physiological data sets completely over the internet continuously.

## 2. System Architecture
The CPS Architecture is divided into structured tiers representing the data pipeline flow:

1. **Hardware / Sensing Layer (Edge):** An Arduino Uno handles an MPU6050 accelerometer and an analog pulse sensor to acquire raw physical data continuously.
2. **Communication Layer:** Hardwired USB Serial connection transferring continuous strings of sensor readings directly to the host PC (Fog Node).
3. **Fog Computing Layer (The Core):** A localized computer or Raspberry Pi receiving the stream. This layer performs:
    - **Preprocessing & Normalization:** Standardizing noise and syncing temporal data frames.
    - **Sliding Window Buffering:** Queuing the incoming stream into `SEQ_LENGTH` (e.g., 30-timestamp) sequence chunks arrays.
    - **LSTM Inference:** Utilizing `tensorflow.keras` applying the deployed multidimensional Long Short-Term Memory machine learning model locally.
    - **Heuristic Computing:** Analyzing variance statistics mapped against neural outputs to generate explainable root-cause alerts over standard error outputs.
4. **Application Dashboard Layer:** A real-time `Streamlit` web application interface rendering local `csv` metrics generated exclusively by the Fog Node.

---
### Setup & Installation
Ensure `Python` is installed on your environment.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Re-train & configure neural models locally
python model/train_lstm.py

# 3. Form the bridge between local ML engines and standard hardware
python fog_node/fog_service.py

# 4. View results directly through a dashboard frontend
streamlit run dashboard/app.py
```
