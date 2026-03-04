import sys
import serial
import time
import numpy as np
import tensorflow as tf
import os
import joblib
import threading
import logging
from collections import deque

# Ensure configuration can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FogNode")


class FogProcessingNode:
    def __init__(self):
        logger.info("Initializing Fog Node processing...")

        # Load model and scaler
        try:
            self.model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
            self.scaler = joblib.load(config.SCALER_PATH)
            logger.info("Model and Scaler loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load model or scaler: {e}")
            self.model = None
            self.scaler = None

        # Thread-safe ring buffer using deque (O(1) append and pop)
        self.data_buffer = deque(maxlen=config.SEQ_LENGTH * 2)
        self.lock = threading.Lock()

        # Initialize output CSV
        os.makedirs(os.path.dirname(config.OUTPUT_FILE), exist_ok=True)
        with open(config.OUTPUT_FILE, 'w') as f:
            f.write("Timestamp,AcX,AcY,AcZ,Pulse,Predicted_State,Confidence,Disturbance_Reason\n")

    def heuristic_analysis(self, window_data):
        """
        Analyze contributing factors to interpret why a disturbance occurred.
        window_data: shape (SEQ_LENGTH, 4) -> AcX, AcY, AcZ, Pulse
        Returns a human-readable reason string.
        """
        ac_data = window_data[:, :3]
        pulse_data = window_data[:, 3]

        ac_std = np.std(ac_data, axis=0)
        movement_intensity = np.mean(ac_std)
        pulse_mean = np.mean(pulse_data)
        pulse_std = np.std(pulse_data)

        # Prioritised check — most severe condition first
        if movement_intensity > 2000:
            return "Excessive Physical Movement"
        elif pulse_mean > 100 and pulse_std > 15:
            return "Abnormal Heart-Rate (Stress/Nightmare)"
        elif movement_intensity > 1000 and pulse_mean > 80:
            return "Restless tossing and turning"
        elif pulse_std > 20:
            return "Irregular Physiological Rhythm"

        return "Normal"

    def _build_feature_vector(self, raw_features):
        """
        Build the 6-feature vector per timestep that matches train_model.py's feature set:
        [movement_magnitude, movement_variance, avg_heart_rate, hrv, movement_frequency, sleep_duration]
        """
        full_features = []
        movements = []
        hrs = []

        for row in raw_features:
            ac_x, ac_y, ac_z, pulse = row
            movement = np.sqrt(ac_x**2 + ac_y**2 + ac_z**2)
            movements.append(movement)
            hrs.append(pulse)

            # HRV approximation: standard deviation of recent heart rate readings
            recent_hrs = hrs[-10:]
            hrv = np.std(recent_hrs) if len(recent_hrs) > 1 else 10.0

            # Rolling statistics (window=10, matching training pipeline)
            recent_movements = movements[-10:]
            movement_variance = np.var(recent_movements) if len(recent_movements) > 1 else 0.0
            avg_heart_rate = np.mean(recent_hrs)
            movement_frequency = sum(1 for m in recent_movements if m > 0.1)
            sleep_duration = 7.0  # Default fixed property for real-time context

            full_features.append([
                movement,
                movement_variance,
                avg_heart_rate,
                hrv,
                movement_frequency,
                sleep_duration
            ])

        return np.array(full_features)

    def process_window(self):
        """
        Processes the sliding window: extract -> build features -> scale -> predict -> save
        """
        with self.lock:
            if len(self.data_buffer) < config.SEQ_LENGTH:
                return
            # Snapshot the latest window (thread-safe copy)
            window = np.array(list(self.data_buffer)[-config.SEQ_LENGTH:])

        timestamps = window[:, 0]
        raw_features = window[:, 1:5].copy()  # [AcX, AcY, AcZ, Pulse]

        if self.model is not None and self.scaler is not None:
            # Build 6-feature matrix matching training pipeline
            full_features = self._build_feature_vector(raw_features)

            # Scale and reshape for LSTM: (1, seq_len, 6)
            scaled_features = self.scaler.transform(full_features)
            input_tensor = scaled_features.reshape(1, config.SEQ_LENGTH, 6)

            # Predict sleep score
            predictions = self.model.predict(input_tensor, verbose=0)
            score = float(predictions[0][0])
            score = np.clip(score, 0, 100)

            # Classification
            if score >= 70:
                state_label = "Good Sleep"
                reason = "Low movement and stable HRV"
            else:
                state_label = "Poor Sleep"
                # Get detailed heuristic reason for poor sleep
                reason = self.heuristic_analysis(raw_features)

            confidence = score / 100.0
        else:
            state_label = "Mock (Model Not Loaded)"
            score = 0.0
            confidence = 0.0
            reason = "None"

        # Write to CSV for dashboard consumption
        latest_timestamp = timestamps[-1]
        latest_acx = raw_features[-1, 0]
        latest_acy = raw_features[-1, 1]
        latest_acz = raw_features[-1, 2]
        latest_pulse = raw_features[-1, 3]

        with open(config.OUTPUT_FILE, 'a') as f:
            f.write(f"{latest_timestamp},{latest_acx},{latest_acy},{latest_acz},"
                    f"{latest_pulse},{state_label},{confidence:.2f},{reason}\n")

        logger.info(f"[{latest_timestamp}] State: {state_label} (Score: {score:.1f}%) | Reason: {reason}")

    def start_mock_stream(self):
        """Simulates a sensor stream for testing without hardware."""
        logger.info("Starting Mock stream (10Hz simulated)...")
        current_time = int(time.time() * 1000)
        while True:
            # Realistic sensor distributions matching training data
            acx = np.random.normal(0.01, 0.05)
            acy = np.random.normal(0.01, 0.05)
            acz = np.random.normal(0.01, 0.05)
            pulse = np.random.normal(65, 5)
            current_time += 100

            with self.lock:
                self.data_buffer.append([current_time, acx, acy, acz, pulse])

            self.process_window()
            time.sleep(1.0 / config.SAMPLING_RATE_HZ)

    def start_serial_stream(self):
        """Reads live sensor data from Arduino via serial port."""
        try:
            ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=1)
            logger.info(f"Connected to Arduino on {config.SERIAL_PORT}")
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 5:
                        try:
                            data = [float(p) for p in parts]
                            with self.lock:
                                self.data_buffer.append(data)

                            if len(self.data_buffer) >= config.SEQ_LENGTH:
                                self.process_window()
                        except ValueError:
                            logger.debug(f"Skipped malformed serial line: {line}")
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            logger.info("Falling back to mock stream...")
            self.start_mock_stream()


if __name__ == "__main__":
    node = FogProcessingNode()
    # Attempts hardware connection; if it fails, switches to mock stream.
    node.start_serial_stream()
