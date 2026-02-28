import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Hardware settings
SERIAL_PORT = 'COM3'   # Arduino Serial Port
BAUD_RATE = 115200

# Fog Processing settings
SEQ_LENGTH = 30        # Number of samples per sliding window (e.g. 30 samples at 10Hz = 3 seconds)
SAMPLING_RATE_HZ = 10

# Model Paths
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'saved_models', 'sleep_lstm_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'saved_models', 'scaler.pkl')

# Data Storage
OUTPUT_FILE = os.path.join(BASE_DIR, 'dashboard', 'live_data.csv')

# Label Mapping for Sleep States
STATE_LABELS = {
    0: "Awake",
    1: "Stable Sleep",
    2: "Restless Sleep",
    3: "Disturbed Sleep"
}

# General UI
DASHBOARD_REFRESH_RATE = 1  # seconds
