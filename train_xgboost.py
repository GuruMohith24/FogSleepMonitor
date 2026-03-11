import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainXGBoost")

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
models_dir = os.path.join(script_dir, "models")
dataset_path = os.path.join(data_dir, "sleep_sensor_dataset.csv")

if not os.path.exists(dataset_path):
    logger.error("Dataset not found. Run prepare_mmash_dataset.py first.")
    exit(1)

# --- 1. Load & Clean Data ---
logger.info(f"Loading data from: {dataset_path}")
data = pd.read_csv(dataset_path)
data = data.dropna()

MAX_ROWS = 100000
if len(data) > MAX_ROWS:
    step = len(data) // MAX_ROWS
    data = data.iloc[::step].head(MAX_ROWS).reset_index(drop=True)

# --- 2. Feature Engineering (Exact matched to LSTM) ---
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

X = data[final_features]
y = data["sleep_score"]

# --- 3. Scale Features ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Build Sequences & Flatten for XGBoost ---
# XGBoost expects 2D tabular data (samples, features).
# We give it the exact same 30-timestep sliding window, but flattened to 180 features (30 * 6)
sequence_length = 30
X_seq, y_seq = [], []

for i in range(len(X_scaled) - sequence_length):
    # Flatten the 30x6 2D array into a 180 1D array
    X_seq.append(X_scaled[i:i + sequence_length].flatten())
    y_seq.append(y.iloc[i + sequence_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
logger.info(f"Created {len(X_seq)} tabular sequences of shape {X_seq.shape[1]} features")

# --- 5. Train/Test Split (Exact matched to LSTM random state) ---
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# --- 6. Train XGBoost ---
logger.info("Training XGBoostRegressor (this might take a minute)...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=7, 
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# --- 7. Evaluate ---
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate precision metrics
acc_10 = np.mean(np.abs(y_pred - y_test) <= 10) * 100
acc_5 = np.mean(np.abs(y_pred - y_test) <= 5) * 100

print("\n" + "="*40)
print("📈 XGBOOST EVALUATION RESULTS")
print("="*40)
print(f"MSE                : {mse:.4f}")
print(f"RMSE               : {rmse:.4f}")
print(f"MAE                : {mae:.4f}")
print(f"R² Score           : {r2:.4f}")
print(f"Accuracy (±10 pts) : {acc_10:.1f}%")
print(f"Accuracy (±5 pts)  : {acc_5:.1f}%")
print("="*40 + "\n")

# --- 8. Save Model ---
model_save_path = os.path.join(models_dir, "xgboost_model.json")
xgb_model.save_model(model_save_path)
logger.info(f"Model saved to: {model_save_path}")
