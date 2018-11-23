
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD 
import numpy as np
import logging
import json

def build_model(num_classes,
                num_bands,
                window_size,
                num_layers = 3,
                num_filters = 64,
                kernel_size = 3,
                pooling_size = 2):

    # create model object
    model = Sequential()
    # is it the first layer
    first_layer = True
    # for each layer
    for _ in range(num_layers):
        # create convolutional layer
        if first_layer:
            conv_layer = Convolution2D(
                num_filters, (kernel_size, kernel_size),
                activation="relu",
                input_shape=(window_size, window_size, num_bands),
                padding="same")
            first_layer = False
        else:
            conv_layer = Convolution2D(
                num_filters, (kernel_size, kernel_size),
                activation="relu",
                padding="same")
        ## add convolutional layer
        model.add(conv_layer)

    # create pooling layer
    pool_layer = MaxPooling2D(
        pool_size=(pooling_size, pooling_size)
        )
    # add the layers to the model
    model.add(pool_layer)

    # add flattening layer
    flat = Flatten()
    model.add(flat)
    # add the dense layer
    dense = Dense(256, activation="relu")
    model.add(dense)
    # add dropout
    drop = Dropout(0.25)
    model.add(drop)
    # add the dense layer
    dense = Dense(128, activation="relu")
    model.add(dense)
    output = Dense(num_classes, activation="softmax")
    model.add(output)
    # compile model
    return model


def train_model(samples, labels, window_size, build_args, train_args):
    
    logger = logging.getLogger(__name__)
    num_classes = labels.shape[1]
    num_samples, Xdim, Ydim, num_bands = samples.shape
    logger.info("Found {} samples with {} bands belonging to {} classes".format(num_samples, num_bands, num_classes))

    ###########
    # Build the Model
    ###########
    model = build_model(num_classes, num_bands, window_size, **build_args)

    ##########
    # Compile the model
    ##########
    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

    ############
    # Train the Model
    ############
    print(labels)
    train_results = model.fit(samples, labels, **train_args)
    json.dump(train_results.history, open("train_results.json", 'w'))

    return model