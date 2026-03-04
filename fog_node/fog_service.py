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
            self.model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
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
            # Extract basic features: AcX, AcY, AcZ, Pulse
            timestamps = window[:, 0]
            raw_features = window[:, 1:5]
            
        if self.model and self.scaler:
            # Construct the 6 features for each timestep matching the trained model 
            full_features = []
            
            # For rolling variance/avg, we'll keep a temporary history context during the window
            movements = []
            hrs = []
            
            for row in raw_features:
                ac_x, ac_y, ac_z, pulse = row
                movement = np.sqrt(ac_x**2 + ac_y**2 + ac_z**2)
                movements.append(movement)
                hrs.append(pulse)
                
                # Mock an HRV based on recent pulse variations (or just a default)
                hrv = max(10, 80 - abs(pulse - 60))
                
                # Compute rolling parameters mapped over previous elements up to window=10
                recent_movements = movements[-10:]
                recent_hrs = hrs[-10:]
                
                movement_variance = np.var(recent_movements) if len(recent_movements) > 1 else 0.0
                avg_heart_rate = np.mean(recent_hrs)
                movement_frequency = sum(1 for m in recent_movements if m > 0.1)
                sleep_duration = 7.0 # Default fixed property 
                
                full_features.append([
                    movement, 
                    movement_variance, 
                    avg_heart_rate, 
                    hrv, 
                    movement_frequency, 
                    sleep_duration
                ])
            
            full_features = np.array(full_features)
            
            # Scale features
            scaled_features = self.scaler.transform(full_features)
            # Reshape for LSTM: (batch, seq_len, features)
            input_tensor = scaled_features.reshape(1, config.SEQ_LENGTH, 6)
            
            # Predict Score
            predictions = self.model.predict(input_tensor, verbose=0)
            score = float(np.squeeze(predictions)[-1] if len(np.squeeze(predictions).shape) > 0 else np.squeeze(predictions))
            
            # Classification
            if score >= 70:
                state_label = "Good Sleep"
                reason = "Low movement and stable HRV"
            else:
                state_label = "Poor Sleep"
                reason = "High movement or unstable HRV"
                
            confidence = score / 100.0 # UI mapping
        else:
            state_label = "Mocking (Model Not Found)"
            confidence = 0.0
            reason = "None"
            
        # Save real-time computation to CSV for Dashboard
        latest_timestamp = timestamps[-1]
        latest_acx = raw_features[-1, 0]
        latest_acy = raw_features[-1, 1]
        latest_acz = raw_features[-1, 2]
        latest_pulse = raw_features[-1, 3]
        
        with open(config.OUTPUT_FILE, 'a') as f:
            f.write(f"{latest_timestamp},{latest_acx},{latest_acy},{latest_acz},{latest_pulse},{state_label},{confidence:.2f},{reason}\n")
            
        logger.info(f"[{latest_timestamp}] State: {state_label} (Score: {score:.1f}%) | Reason: {reason}")
        
    def start_mock_stream(self):
        """ Mocks a serial stream for testing purposes without the hardware. """
        logger.info("Starting Mock stream (10Hz simulated)...")
        current_time = int(time.time() * 1000)
        while True:
            # generate mock data matching real distributions (-0.2 to 0.2 ac, 50 to 90 pulse)
            acx = np.random.normal(0.01, 0.05)
            acy = np.random.normal(0.01, 0.05)
            acz = np.random.normal(0.01, 0.05)
            pulse = np.random.normal(65, 5)
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
