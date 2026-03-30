import os
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TFLiteConverter")

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")
keras_model_path = os.path.join(models_dir, "sleep_lstm_model.h5")
tflite_model_path = os.path.join(models_dir, "sleep_model.tflite")

if not os.path.exists(keras_model_path):
    logger.error(f"Cannot find Keras model at {keras_model_path}")
    logger.error("Run 'train_model.py' first.")
    exit(1)

logger.info(f"Loading heavy Keras model from {keras_model_path}...")
try:
    keras_model = tf.keras.models.load_model(keras_model_path, compile=False)
    
    # --- Hack for LSTM TFLite Conversion ---
    # TFLite hates dynamic batch sizes (None) for LSTMs. It uses FlexOps which crash on Windows/Microcontrollers.
    # By forcing a static batch shape of 1, TFLite can natively compile it to clean C++ instructions.
    
    # Create an identical model with a static batch size
    input_shape = (1, 30, 6)  # (batch_size, seq_length, features)
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape[1:], batch_size=input_shape[0]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    model.set_weights(keras_model.get_weights())
    
    logger.info("Model loaded and statically shaped for TFLite successfully.")
    
    # --- Convert to TFLite ---
    logger.info("Converting model to TensorFlow Lite format (Native Ops only)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations for size and speed (Default quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
        
    # Get file sizes to show the difference
    original_size = os.path.getsize(keras_model_path) / 1024
    tflite_size = os.path.getsize(tflite_model_path) / 1024
    
    logger.info(f"SUCCESS! TFLite model saved to {tflite_model_path}")
    logger.info(f"Size Reduction: {original_size:.1f} KB ---> {tflite_size:.1f} KB")

except Exception as e:
    logger.error(f"Conversion failed: {e}")
