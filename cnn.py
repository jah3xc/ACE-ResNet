
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
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(window_size, window_size, num_bands))
    )
    model.add(tf.keras.layers.Activation("relu"))
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3))
    )
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation("softmax"))

    return model


def train_model(samples, labels, window_size, train_args):

    logger = logging.getLogger(__name__)
    num_classes = labels.shape[1]
    num_samples, Xdim, Ydim, num_bands = samples.shape
    logger.info("Found {} samples with {} bands belonging to {} classes".format(num_samples, num_bands, num_classes))

    ###########
    # Build the Model
    ###########
    model = build_model(num_classes, num_bands, window_size)

    ##########
    # Compile the model
    ##########
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    ############
    # Train the Model
    ############
    model.fit(samples, labels, **train_args)

    return model