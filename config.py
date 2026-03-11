import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Hardware Settings ---
SERIAL_PORT = os.environ.get('SLEEP_SERIAL_PORT', 'COM3')  # Override via env variable
BAUD_RATE = 115200

# --- Fog Processing Settings ---
SEQ_LENGTH = 30          # Samples per sliding window (30 samples at 10Hz = 3 seconds)
SAMPLING_RATE_HZ = 10

# --- Model Paths ---
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sleep_lstm_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# --- Data Storage ---
OUTPUT_FILE = os.path.join(BASE_DIR, 'dashboard', 'live_data.csv')

# --- Label Mapping for Sleep States ---
STATE_LABELS = {
    0: "Good Sleep",
    1: "Poor Sleep"
}

# --- Dashboard UI ---
DASHBOARD_REFRESH_RATE = 1  # seconds
