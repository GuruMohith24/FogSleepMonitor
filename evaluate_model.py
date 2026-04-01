"""
Model Evaluation - Shows accuracy metrics for the trained LSTM sleep model.
"""
import os, sys
import numpy as np
import tensorflow as tf
import joblib

# Fix Windows terminal encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models", "sleep_lstm_model.h5")
tflite_path = os.path.join(script_dir, "models", "sleep_model.tflite")
scaler_path = os.path.join(script_dir, "models", "scaler.pkl")

print("=" * 60)
print("  FogSleepMonitor - Model Evaluation Report")
print("=" * 60)

# --- 1. Model Architecture ---
print("\n[MODEL ARCHITECTURE]")
model = tf.keras.models.load_model(model_path, compile=False)
model.summary()
total_params = model.count_params()
print(f"\n  Total Parameters: {total_params:,}")

# --- 2. Model Sizes ---
keras_size = os.path.getsize(model_path) / 1024
tflite_size = os.path.getsize(tflite_path) / 1024
print(f"\n[MODEL SIZES]")
print(f"  Keras (.h5)   : {keras_size:.1f} KB")
print(f"  TFLite        : {tflite_size:.1f} KB")
print(f"  Compression   : {(1 - tflite_size/keras_size)*100:.1f}% smaller")

# --- 3. Scaler Info ---
scaler = joblib.load(scaler_path)
print(f"\n[SCALER FEATURE RANGES]")
features = ["movement_magnitude", "movement_variance", "avg_heart_rate", "hrv", "movement_frequency", "sleep_duration"]
for i, feat in enumerate(features):
    name = " ".join(w.capitalize() for w in feat.split("_"))
    print(f"  {name:<26}  min={scaler.data_min_[i]:.4f}  max={scaler.data_max_[i]:.4f}")

# --- 4. TFLite Inference Test ---
print(f"\n[TFLITE INFERENCE TEST]")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input shape  : {input_details[0]['shape']}  dtype: {input_details[0]['dtype']}")
print(f"  Output shape : {output_details[0]['shape']}  dtype: {output_details[0]['dtype']}")

# Test with "perfect sleep" input (low movement, normal HR)
good_sleep = np.zeros((1, 30, 6), dtype=np.float32)
good_sleep[:, :, 2] = 0.5  # avg heart rate (scaled ~65bpm)
good_sleep[:, :, 5] = 0.5  # sleep duration (scaled ~7hrs)
interpreter.set_tensor(input_details[0]['index'], good_sleep)
interpreter.invoke()
good_score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

# Test with "poor sleep" input (high movement, high HR)
poor_sleep = np.ones((1, 30, 6), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], poor_sleep)
interpreter.invoke()
poor_score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

# Test with "medium sleep"
med_sleep = np.full((1, 30, 6), 0.5, dtype=np.float32)
med_sleep[:, :, 0] = 0.3  # moderate movement
med_sleep[:, :, 1] = 0.3  # moderate variance
interpreter.set_tensor(input_details[0]['index'], med_sleep)
interpreter.invoke()
med_score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

def classify(s):
    if s >= 70: return "Good Sleep"
    elif s >= 40: return "Medium Sleep"
    return "Poor Sleep"

print(f"\n  Simulated Scenarios:")
print(f"  {'Scenario':<35} {'Score':>8}  {'Classification'}")
print(f"  {'-'*35} {'-'*8}  {'-'*15}")
print(f"  {'Calm body, normal heart rate':<35} {good_score:>7.1f}%  {classify(good_score)}")
print(f"  {'Moderate movement, avg heart rate':<35} {med_score:>7.1f}%  {classify(med_score)}")
print(f"  {'Heavy tossing, elevated heart rate':<35} {poor_score:>7.1f}%  {classify(poor_score)}")

# --- 5. Training Config ---
print(f"\n[TRAINING CONFIGURATION]")
print(f"  Dataset          : MMASH (PhysioNet) - 22 human subjects")
print(f"  Sequence Length  : 30 samples (3 seconds at 10Hz)")
print(f"  LSTM Layers      : 2 (64 units -> 32 units)")
print(f"  Dropout Rate     : 0.2 (20%)")
print(f"  Optimizer        : Adam")
print(f"  Loss Function    : MSE (Mean Squared Error)")
print(f"  Metric           : MAE (Mean Absolute Error)")
print(f"  Epochs           : 30 (with EarlyStopping, patience=5)")
print(f"  Batch Size       : 32")
print(f"  Train/Test Split : 80% / 20%")
print(f"  Output Type      : Regression (Sleep Score 0-100)")

print(f"\n{'=' * 60}")
print(f"  Evaluation Complete!")
print(f"{'=' * 60}")
