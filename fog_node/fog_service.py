import sys
import serial
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib
import threading
import logging

# Ensure configuration can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FogNode")

class FogProcessingNode:
    def __init__(self):
        logger.info("Initializing Fog Node processing...")
        # Try loading model and scaler
        try:
            self.model = tf.keras.models.load_model(config.MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
            logger.info("Model and Scaler loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load model or scaler. {e}")
            self.model = None
            self.scaler = None
            
        self.data_buffer = []
        self.lock = threading.Lock()
        
        # Initialize output CSV
        os.makedirs(os.path.dirname(config.OUTPUT_FILE), exist_ok=True)
        with open(config.OUTPUT_FILE, 'w') as f:
            f.write("Timestamp,AcX,AcY,AcZ,Pulse,Predicted_State,Confidence,Disturbance_Reason\n")
            
    def heuristic_analysis(self, window_data):
        """
        Analyze contributing factors to interpret why a disturbance occurred.
        window_data: shape (SEQ_LENGTH, 4) -> AcX, AcY, AcZ, Pulse
        """
        ac_data = window_data[:, :3]
        pulse_data = window_data[:, 3]
        
        ac_std = np.std(ac_data, axis=0) # movement variations
        pulse_mean = np.mean(pulse_data)
        pulse_std = np.std(pulse_data)
        
        reason = "Normal"
        
        if np.mean(ac_std) > 2000:
            reason = "Excessive Physical Movement"
        elif pulse_mean > 620 and pulse_std > 80:
            reason = "Abnormal Heart-Rate (Stress/Nightmare)"
        elif pulse_std > 120:
            reason = "Irregular Physiological Rhythm"
        elif np.mean(ac_std) > 1000 and pulse_mean > 600:
            reason = "Restless tossing and turning"
            
        return reason

    def process_window(self):
        """
        Processes the sliding window: scale -> predict -> heuristic -> save
        """
        with self.lock:
            if len(self.data_buffer) < config.SEQ_LENGTH:
                return
            # Get latest SEQ_LENGTH samples
            window = np.array(self.data_buffer[-config.SEQ_LENGTH:])
            # Extract just the features, leaving out timestamp for model
            timestamps = window[:, 0]
            features = window[:, 1:5]
            
        if self.model and self.scaler:
            # Scale features
            scaled_features = self.scaler.transform(features)
            # Reshape for LSTM: (batch, seq_len, features)
            input_tensor = scaled_features.reshape(1, config.SEQ_LENGTH, 4)
            
            # Predict
            predictions = self.model.predict(input_tensor, verbose=0)
            state_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            state_label = config.STATE_LABELS.get(state_idx, "Unknown")
            
            # Heuristic interpretation
            if state_idx in [2, 3]: # Restless or Disturbed
                reason = self.heuristic_analysis(features)
            else:
                reason = "None"
        else:
            state_label = "Mocking (Model Not Found)"
            confidence = 0.0
            reason = "None"
            
        # Save real-time computation to CSV for Dashboard
        latest_timestamp = timestamps[-1]
        latest_acx = features[-1, 0]
        latest_acy = features[-1, 1]
        latest_acz = features[-1, 2]
        latest_pulse = features[-1, 3]
        
        with open(config.OUTPUT_FILE, 'a') as f:
            f.write(f"{latest_timestamp},{latest_acx},{latest_acy},{latest_acz},{latest_pulse},{state_label},{confidence:.2f},{reason}\n")
            
        logger.info(f"[{latest_timestamp}] State: {state_label} (Conf: {confidence:.2f}) | Reason: {reason}")
        
    def start_mock_stream(self):
        """ Mocks a serial stream for testing purposes without the hardware. """
        logger.info("Starting Mock stream (10Hz simulated)...")
        current_time = int(time.time() * 1000)
        while True:
            # generate mock data similar to awake/sleep
            acx = np.random.randint(-1500, 1500)
            acy = np.random.randint(-1500, 1500)
            acz = np.random.randint(-1500, 1500)
            pulse = np.random.randint(500, 700)
            current_time += 100
            
            with self.lock:
                self.data_buffer.append([current_time, acx, acy, acz, pulse])
                if len(self.data_buffer) > config.SEQ_LENGTH * 2:
                    self.data_buffer.pop(0)
            
            self.process_window()
            time.sleep(1.0 / config.SAMPLING_RATE_HZ)
            
    def start_serial_stream(self):
        try:
            ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=1)
            logger.info(f"Connected to Arduino on {config.SERIAL_PORT}")
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 5:
                        try:
                            # Parse data: Timestamp, AcX, AcY, AcZ, PulseValue
                            data = [float(p) for p in parts]
                            with self.lock:
                                self.data_buffer.append(data)
                                if len(self.data_buffer) > config.SEQ_LENGTH * 2:
                                    self.data_buffer.pop(0)

                            if len(self.data_buffer) >= config.SEQ_LENGTH:
                                self.process_window()
                        except ValueError:
                            pass
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            logger.info("Falling back to mock stream...")
            self.start_mock_stream()

if __name__ == "__main__":
    node = FogProcessingNode()
    # Attempts hardware connection; if fails, does Mock stream.
    node.start_serial_stream()
