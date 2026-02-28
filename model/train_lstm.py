import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
import joblib
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainLSTM")

# Ensure reproducible synthetic dataset
np.random.seed(42)

def generate_synthetic_data(num_samples=10000, seq_length=30):
    logger.info("Generating synthetic sleep sensor dataset...")
    # Generate continuous variables: AcX, AcY, AcZ, Pulse
    # Sleep states: 0 = Awake, 1 = Stable Sleep, 2 = Restless Sleep, 3 = Disturbed Sleep
    
    X = []
    y = []
    
    for i in range(num_samples):
        state = np.random.choice([0, 1, 2, 3], p=[0.1, 0.6, 0.2, 0.1])
        
        if state == 0: # Awake: High movement, high HR variability
            ac = np.random.normal(loc=2000, scale=3000, size=(seq_length, 3))
            pulse = np.random.normal(loc=600, scale=100, size=(seq_length, 1))
        elif state == 1: # Stable: Low movement, steady low HR
            ac = np.random.normal(loc=0, scale=300, size=(seq_length, 3))
            pulse = np.random.normal(loc=512, scale=20, size=(seq_length, 1))
        elif state == 2: # Restless: Occasional large movements, unsteady HR
            ac = np.random.normal(loc=500, scale=1500, size=(seq_length, 3))
            pulse = np.random.normal(loc=550, scale=60, size=(seq_length, 1))
        elif state == 3: # Disturbed: Sudden spikes in HR and movement
            ac = np.random.normal(loc=1000, scale=4000, size=(seq_length, 3))
            pulse = np.random.normal(loc=650, scale=150, size=(seq_length, 1))
        
        sequence = np.hstack([ac, pulse])
        X.append(sequence)
        y.append(state)
        
    return np.array(X), np.array(y)

def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    SEQ_LENGTH = 30 # Window size (e.g. 3 seconds of 10Hz data)
    NUM_CLASSES = 4
    
    # 1. Generate & prepare Data
    X, y = generate_synthetic_data(num_samples=3000, seq_length=SEQ_LENGTH)
    
    # We must scale the features
    # Reshape to 2D for the scaler, then back to 3D
    num_samples, seq_len, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(num_samples, seq_len, num_features)
    
    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Build & Train Model
    logger.info("Building LSTM model...")
    model = build_model((SEQ_LENGTH, num_features), NUM_CLASSES)
    model.summary(print_fn=logger.info)
    
    logger.info("Training model...")
    # Train less epochs to save time since it's just meant as a structural implementation
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # 4. Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    logger.info(f"Test Accuracy: {acc:.4f} | Validation Loss: {loss:.4f}")
    
    # 5. Save model and scaler
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    model.save(os.path.join(model_dir, "sleep_lstm_model.h5"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    logger.info("Model and Scaler saved successfully in 'saved_models/'!")
