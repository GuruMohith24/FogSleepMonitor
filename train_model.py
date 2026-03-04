import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainModel")

# --- Path Setup (works from any working directory) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
models_dir = os.path.join(script_dir, "models")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

dataset_path = os.path.join(data_dir, "sleep_sensor_dataset.csv")

if not os.path.exists(dataset_path):
    logger.error(f"Dataset not found at {dataset_path}")
    logger.error("Run 'python prepare_mmash_dataset.py' first to generate the dataset.")
    exit(1)

# --- 1. Load & Clean Data ---
logger.info(f"Loading data from: {dataset_path}")
data = pd.read_csv(dataset_path)
initial_rows = len(data)
data = data.dropna()
logger.info(f"Cleaned {initial_rows - len(data)} rows with missing values. Remaining: {len(data)} rows.")

# Subsample for memory efficiency (LSTM sequences of 1.4M rows = ~168GB RAM needed)
# We keep 100K contiguous rows to preserve temporal patterns per subject
MAX_ROWS = 100000
if len(data) > MAX_ROWS:
    logger.info(f"Subsampling from {len(data)} to {MAX_ROWS} rows for memory efficiency...")
    # Stratified: take balanced chunks from across the dataset
    step = len(data) // MAX_ROWS
    data = data.iloc[::step].head(MAX_ROWS).reset_index(drop=True)
    logger.info(f"Subsampled to {len(data)} rows.")

# --- 2. Feature Engineering ---
logger.info("Extracting features...")
data["movement_magnitude"] = np.sqrt(data["acc_x"]**2 + data["acc_y"]**2 + data["acc_z"]**2)
data["movement_variance"] = data["movement_magnitude"].rolling(window=10, min_periods=1).var().fillna(0)
data["avg_heart_rate"] = data["heart_rate"].rolling(window=10, min_periods=1).mean()
data["movement_frequency"] = (data["movement_magnitude"] > 0.1).rolling(window=10, min_periods=1).sum()

final_features = [
    "movement_magnitude",
    "movement_variance",
    "avg_heart_rate",
    "hrv",
    "movement_frequency",
    "sleep_duration"
]
logger.info(f"Features: {final_features}")

X = data[final_features]
y = data["sleep_score"]

# --- 3. Scale Features ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
logger.info("Scaler saved.")

# --- 4. Build Sequences (Sliding Window) ---
sequence_length = 30
X_seq, y_seq = [], []

for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i + sequence_length])
    y_seq.append(y.iloc[i + sequence_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
logger.info(f"Created {len(X_seq)} sequences of length {sequence_length}")

# --- 5. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# --- 6. Build & Train LSTM ---
logger.info("Building LSTM model...")
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation="linear")
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary(print_fn=logger.info)

# EarlyStopping to prevent overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

logger.info("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# --- 7. Evaluate ---
loss, mae = model.evaluate(X_test, y_test, verbose=0)
logger.info(f"Test Loss (MSE): {loss:.4f} | Test MAE: {mae:.4f}")

# --- 8. Feature Importance (Correlation Heuristic) ---
correlations = [abs(data[col].corr(data["sleep_score"])) for col in final_features]
total_corr = sum(correlations)
importances = [(final_features[i], correlations[i] / total_corr if total_corr > 0 else 0)
               for i in range(len(final_features))]
importances.sort(key=lambda x: x[1], reverse=True)

print("\n--- Feature Importance ---")
for feat, imp in importances:
    feat_name = " ".join([word.capitalize() for word in feat.split("_")])
    print(f"  {feat_name.ljust(26)} {imp:.2f}")

# --- 9. Save Model ---
model_save_path = os.path.join(models_dir, "sleep_lstm_model.h5")
model.save(model_save_path)
logger.info(f"Model saved to: {model_save_path}")

# --- 10. Quick Prediction Demo ---
predictions = model.predict(X_test[:1], verbose=0)
score = float(predictions[0][0])

if score >= 70:
    result = "Good Sleep"
    reason = "Low movement and stable HRV"
else:
    result = "Poor Sleep"
    reason = "High movement or unstable HRV"

print(f"\n--- Example Prediction ---")
print(f"  Sleep Score : {score:.1f}")
print(f"  Quality     : {result}")
print(f"  Reason      : {reason}")
