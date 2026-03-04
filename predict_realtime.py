import os
import numpy as np
import tensorflow as tf
import joblib

models_dir = "models"
model_path = os.path.join(models_dir, "sleep_lstm_model.h5")
scaler_path = os.path.join(models_dir, "scaler.pkl")

print("Loading Model and Scaler...")
model = tf.keras.models.load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)

# Mocked history variables
history_buffer = []

def predict_sleep(acc_x, acc_y, acc_z, heart_rate, hrv):
    global history_buffer
    
    movement = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Store history for rolling calculations
    history_buffer.append({"movement": movement, "hr": heart_rate})
    if len(history_buffer) > 10:
        history_buffer.pop(0)
    
    # Rolling calculations
    movements = [x["movement"] for x in history_buffer]
    hrs = [x["hr"] for x in history_buffer]
    
    movement_variance = np.var(movements) if len(movements) > 1 else 0.0
    avg_heart_rate = np.mean(hrs)
    movement_frequency = sum(1 for m in movements if m > 0.1)
    sleep_duration = 7.0 # Default fixed duration
    
    # Needs sequence of 6 features: ["movement_magnitude", "movement_variance", "avg_heart_rate", "hrv", "movement_frequency", "sleep_duration"]
    sample = [[movement, movement_variance, avg_heart_rate, hrv, movement_frequency, sleep_duration]]
    
    # Transform using scaler
    sample_scaled = scaler.transform(sample)
    
    # Replicate to fit sequence dimension of 30 if model expects window
    # Shape expected: (1, 30, 6)
    sample_seq = np.repeat(sample_scaled, 30, axis=0).reshape(1, 30, 6)
    
    score_pred = model.predict(sample_seq, verbose=0)
    
    # Extract
    score = float(np.squeeze(score_pred)[-1] if len(np.squeeze(score_pred).shape) > 0 else np.squeeze(score_pred))
    
    if score >= 70:
        label = "Good Sleep"
        reason = "Low movement and stable HRV"
    else:
        label = "Poor Sleep"
        reason = "High movement or unstable HRV"
        
    return score, label, reason

if __name__ == "__main__":
    test_acc_x = 0.02
    test_acc_y = 0.01
    test_acc_z = 0.03
    test_heart_rate = 62
    test_hrv = 45
    
    # Call multiple times to build history
    for _ in range(5):
        score, label, reason = predict_sleep(test_acc_x, test_acc_y, test_acc_z, test_heart_rate, test_hrv)
        
    print("\nSleep Score:", int(score))
    print("Classification:", label)
    print("Reason:", reason)
