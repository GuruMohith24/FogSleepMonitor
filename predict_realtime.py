import os
import numpy as np
import tensorflow as tf
import joblib
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictRealtime")

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")
model_path = os.path.join(models_dir, "sleep_lstm_model.h5")
scaler_path = os.path.join(models_dir, "scaler.pkl")

logger.info("Loading Model and Scaler...")
model = tf.keras.models.load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)
logger.info("Model and Scaler loaded.")

# Sequence length must match training config
SEQUENCE_LENGTH = 30

# Rolling buffer for building proper sequences (not fake padding)
history_buffer = deque(maxlen=SEQUENCE_LENGTH)


def predict_sleep(acc_x, acc_y, acc_z, heart_rate, hrv):
    """
    Accepts a single sensor reading, appends to the rolling buffer,
    and returns a prediction once enough history is available.

    Returns: (score, label, reason) or None if not enough history yet.
    """
    movement = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # Build rolling statistics from history
    past_movements = [h["movement"] for h in history_buffer]
    past_hrs = [h["hr"] for h in history_buffer]

    past_movements.append(movement)
    past_hrs.append(heart_rate)

    recent_movements = past_movements[-10:]
    recent_hrs = past_hrs[-10:]

    movement_variance = np.var(recent_movements) if len(recent_movements) > 1 else 0.0
    avg_heart_rate = np.mean(recent_hrs)
    movement_frequency = sum(1 for m in recent_movements if m > 0.1)
    hrv_computed = np.std(recent_hrs) if len(recent_hrs) > 1 else hrv
    sleep_duration = 7.0  # Default for real-time context

    # Feature vector: matches train_model.py's feature set exactly
    feature_row = [movement, movement_variance, avg_heart_rate, hrv_computed,
                   movement_frequency, sleep_duration]

    # Append to rolling buffer
    history_buffer.append({"movement": movement, "hr": heart_rate, "features": feature_row})

    # Need at least SEQUENCE_LENGTH samples for a valid prediction
    if len(history_buffer) < SEQUENCE_LENGTH:
        return None

    # Build the sequence from the rolling buffer (proper temporal data!)
    sequence = np.array([h["features"] for h in history_buffer])

    # Scale and reshape for LSTM: (1, 30, 6)
    scaled = scaler.transform(sequence)
    input_tensor = scaled.reshape(1, SEQUENCE_LENGTH, 6)

    # Predict
    prediction = model.predict(input_tensor, verbose=0)
    score = float(prediction[0][0])
    score = np.clip(score, 0, 100)

    if score >= 70:
        label = "Good Sleep"
        reason = "Low movement and stable HRV"
    else:
        label = "Poor Sleep"
        reason = "High movement or unstable HRV"

    return score, label, reason


if __name__ == "__main__":
    # Simulate multiple readings to build up history
    test_samples = [
        (0.02, 0.01, 0.03, 62, 45),
        (0.01, 0.02, 0.01, 64, 42),
        (0.03, 0.01, 0.02, 63, 48),
        (0.01, 0.03, 0.01, 61, 44),
        (0.02, 0.02, 0.02, 65, 46),
    ]

    print("Building sensor history...\n")

    # Feed enough samples to fill the buffer
    for i in range(SEQUENCE_LENGTH + 5):
        sample = test_samples[i % len(test_samples)]
        result = predict_sleep(*sample)

        if result is None:
            print(f"  Sample {i+1:3d}/{SEQUENCE_LENGTH}: buffering...")
        else:
            score, label, reason = result
            print(f"  Sample {i+1:3d}: Score={score:.1f}  |  {label}")

    # Final result
    if result is not None:
        score, label, reason = result
        print(f"\n--- Final Prediction ---")
        print(f"  Sleep Score   : {score:.1f}")
        print(f"  Classification: {label}")
        print(f"  Reason        : {reason}")
    else:
        print("\nNot enough data for prediction yet.")
