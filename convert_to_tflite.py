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
    model = tf.keras.models.load_model(keras_model_path, compile=False)
    logger.info("Model loaded successfully.")
    
    # --- Convert to TFLite ---
    logger.info("Converting model to TensorFlow Lite format (Optimizing for Edge)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations for size and speed (Default quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Note: LSTM layers require Select TF Ops or specific Experimental Lowering
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_lower_tensor_list_ops = False
    
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
