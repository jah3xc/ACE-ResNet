import tensorflow as tf

def build_model():
    model = tf.keras.Sequential()
    conv = tf.layers.Conv2D(32, (3, 3))
    model.add(conv)
    return model

def train_model(samples, labels):
    model = build_model()