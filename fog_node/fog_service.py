import serial
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib
import threading

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERIAL_PORT = 'COM3' # Change this depending on Arduino's port
BAUD_RATE = 115200
SEQ_LENGTH = 30
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'saved_models', 'sleep_lstm_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'saved_models', 'scaler.pkl')
OUTPUT_FILE = os.path.join(BASE_DIR, 'dashboard', 'live_data.csv')

# States mapping
STATE_LABELS = {
    0: "Awake",
    1: "Stable Sleep",
    2: "Restless Sleep",
    3: "Disturbed Sleep"
}

class FogProcessingNode:
    def __init__(self):
        print("Initializing Fog Node processing...")
        # Try loading model and scaler
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("Model and Scaler loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load model or scaler. {e}")
            self.model = None
            self.scaler = None
            
        self.data_buffer = []
        self.lock = threading.Lock()
        
        # Initialize output CSV
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            f.write("Timestamp,AcX,AcY,AcZ,Pulse,Predicted_State,Disturbance_Reason\n")
            
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
            if len(self.data_buffer) < SEQ_LENGTH:
                return
            # Get latest SEQ_LENGTH samples
            window = np.array(self.data_buffer[-SEQ_LENGTH:])
            # Extract just the features, leaving out timestamp for model
            # window elements: [timestamp, AcX, AcY, AcZ, Pulse]
            timestamps = window[:, 0]
            features = window[:, 1:5]
            
        if self.model and self.scaler:
            # Scale features
            scaled_features = self.scaler.transform(features)
            # Reshape for LSTM: (batch, seq_len, features)
            input_tensor = scaled_features.reshape(1, SEQ_LENGTH, 4)
            
            # Predict
            predictions = self.model.predict(input_tensor, verbose=0)
            state_idx = np.argmax(predictions[0])
            state_label = STATE_LABELS.get(state_idx, "Unknown")
            
            # Heuristic interpretation
            if state_idx in [2, 3]: # Restless or Disturbed
                reason = self.heuristic_analysis(features)
            else:
                reason = "None"
        else:
            state_label = "Mocking (Model Not Found)"
            reason = "None"
            
        # Save real-time computation to CSV for Dashboard
        latest_timestamp = timestamps[-1]
        latest_acx = features[-1, 0]
        latest_acy = features[-1, 1]
        latest_acz = features[-1, 2]
        latest_pulse = features[-1, 3]
        
        with open(OUTPUT_FILE, 'a') as f:
            f.write(f"{latest_timestamp},{latest_acx},{latest_acy},{latest_acz},{latest_pulse},{state_label},{reason}\n")
            
        print(f"[{latest_timestamp}] State: {state_label} | Reason: {reason}")
        
    def start_mock_stream(self):
        """ Mocks a serial stream for testing purposes without the hardware. """
        print("Starting Mock stream...")
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
                if len(self.data_buffer) > SEQ_LENGTH * 2:
                    self.data_buffer.pop(0)
            
            self.process_window()
            time.sleep(0.1) # 10Hz
            
    def start_serial_stream(self):
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Connected to Arduino on {SERIAL_PORT}")
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
                                if len(self.data_buffer) > SEQ_LENGTH * 2:
                                    self.data_buffer.pop(0)

                            if len(self.data_buffer) >= SEQ_LENGTH:
                                self.process_window()
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Serial connection failed: {e}")
            print("Falling back to mock stream...")
            self.start_mock_stream()

if __name__ == "__main__":
    node = FogProcessingNode()
    # Attempts hardware connection; if fails, does Mock stream.
    node.start_serial_stream()
