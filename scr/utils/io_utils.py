import tensorflow as tf

def load_model(model_path:str):
    return tf.keras.models.load_model(model_path)