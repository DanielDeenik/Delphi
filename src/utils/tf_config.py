
import tensorflow as tf
import os

def configure_tensorflow():
    # Enable memory growth to prevent TF from taking all GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    # Use mixed precision for faster computation and less memory
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Set inter/intra op parallelism for CPU optimization
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    return physical_devices
