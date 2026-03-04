import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

data_dir = "data"
models_dir = "models"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

dataset_path = os.path.join(data_dir, "sleep_sensor_dataset.csv")

if not os.path.exists(dataset_path):
    print("Dataset not found, generating synthetic data matching exact structure...")
    num_samples = 5000
    timestamps = pd.date_range("2023-01-01", periods=num_samples, freq="1S").strftime("%H:%M:%S")
    acc_x = np.random.normal(0.05, 0.1, num_samples)
    acc_y = np.random.normal(0.05, 0.1, num_samples)
    acc_z = np.random.normal(0.05, 0.1, num_samples)
    heart_rate = np.random.normal(65, 10, num_samples)
    hrv = np.random.normal(45, 15, num_samples)
    sleep_duration = np.random.normal(7.0, 1.5, num_samples)
    
    movement = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    movement_level = ["high" if m > 0.3 else "low" for m in movement]
    
    # Fake score
    sleep_score = 100 - (movement * 100) - ((heart_rate - 60) * 0.5)
    sleep_score = np.clip(sleep_score, 0, 100)
    sleep_quality = ["good" if s >= 70 else "poor" for s in sleep_score]
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "heart_rate": heart_rate,
        "hrv": hrv,
        "sleep_duration": sleep_duration,
        "movement_level": movement_level,
        "sleep_quality": sleep_quality,
        "sleep_score": sleep_score
    })
    df.to_csv(dataset_path, index=False)

print(f"Loaded data from: {dataset_path}")
print("\nCleaning dataset...")
data = pd.read_csv(dataset_path)
print("Removing missing values...")
data = data.dropna()

print("Normalizing signals...")

print("\nExtracting features...")
# User defined features:
data["movement_magnitude"] = np.sqrt(data["acc_x"]**2 + data["acc_y"]**2 + data["acc_z"]**2)

# Compute additional features for importance (rolling variance / avg for the window context)
data["movement_variance"] = data["movement_magnitude"].rolling(window=10, min_periods=1).var().fillna(0)
data["avg_heart_rate"] = data["heart_rate"].rolling(window=10, min_periods=1).mean()
data["movement_frequency"] = (data["movement_magnitude"] > 0.1).rolling(window=10, min_periods=1).sum()

# We will use the features exactly as defined in 'Final feature set example' plus original
final_features = [
    "movement_magnitude",
    "movement_variance",
    "avg_heart_rate",
    "hrv",
    "movement_frequency",
    "sleep_duration"
]
for f in final_features:
    print(f)

X = data[final_features]
# The user wants "sleep_score" as target for LSTM regression
y = data["sleep_score"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

sequence_length = 30
X_seq, y_seq = [], []

for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i + sequence_length])
    y_seq.append(y.iloc[i + sequence_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

print("\nTraining LSTM model...")
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Feature Importance calculation for LSTM (Using correlation heuristic to rank influences)
# We find absolute correlation of each feature against the target to emulate feature importance
correlations = [abs(data[col].corr(data["sleep_score"])) for col in final_features]
total_corr = sum(correlations)
importances = [(final_features[i], correlations[i] / total_corr if total_corr > 0 else 0) for i in range(len(final_features))]
importances.sort(key=lambda x: x[1], reverse=True)

print("\n--- Feature Importance ---")
for feat, imp in importances:
    # Formatting for aesthetic printing (e.g. Camel Case representation)
    feat_name = " ".join([word.capitalize() for word in feat.split("_")])
    print(f"{feat_name.ljust(26)} {imp:.2f}")

model_save_path = os.path.join(models_dir, "sleep_lstm_model.h5")
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# Predict Sleep Score
predictions = model.predict(X_test)
score = predictions[0][-1][0] if len(predictions.shape) == 3 else predictions[0][0]

if score >= 70:
    result = "Good Sleep"
else:
    result = "Poor Sleep"

print("\nExample Final System Output")
print(f"Sleep Score: {score:.1f}")
print(f"Classification: {result}")
if result == "Good Sleep":
    print("Reason: Low movement and stable HRV")
else:
    print("Reason: High movement or unstable HRV")
