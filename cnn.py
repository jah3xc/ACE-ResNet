
import tensorflow as tf
import numpy as np
import logging

def build_model(num_classes, num_bands, window_size):
    #########
    # define sequential model
    ##########
    model = tf.keras.Sequential()

    ##########
    # Define Architecture
    ##########
    model.add(tf.layers.Conv2D(32, (3, 3), input_shape=(window_size, window_size, num_bands)))
    # model.add(tf.layers.Conv2D(32, (3, 3)))
    # model.add(tf.layers.MaxPooling2D((2,2), 2)
    model.add(tf.keras.layers.Flatten())
    # model.add(tf.layers.Dense(64, activation="relu"))
    model.add(tf.layers.Dense(num_classes, activation="softmax"))

    return model


def train_model(samples, labels, window_size, train_args):

    logger = logging.getLogger(__name__)
    num_classes = len(np.unique(labels))
    num_samples, Xdim, Ydim, num_bands = samples.shape
    logger.debug("Found {} bands".format(num_bands))

    ###########
    # Build the Model
    ###########
    model = build_model(num_classes, num_bands, window_size)

    ##########
    # Compile the model
    ##########
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    ############
    # Train the Model
    ############
    model.fit(samples, labels)

    return model