import tensorflow as tf
import platform
import sys

print("=== DIAGNOSA VERSI ===")
print(f"Python Version    : {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")

try:
    import keras
    print(f"Keras Version     : {keras.__version__}")
except ImportError:
    print("Keras Version     : (Not installed separately)")

print("\nCek Keras Internal:")
print(f"tf.keras Version  : {tf.keras.__version__}")
print("======================")
