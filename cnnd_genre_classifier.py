# import what we need and set up globals

import os
import librosa
import math
import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras 

DATASET_PATH = "genres_original"
JSON_PATH = "data\GTZAN_dict.json"

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    Y = np.array(data["labels"])

def prepare_datasets(my_test_size, validation_size):
    # load the data
    X, Y = load_data(DATASET_PATH)

    # create the train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=my_test_size)
    
    # create the validation test split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

def build_model(my_input_shape):

    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu"), input_shape = my_input_shape)
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"),
              input_shape=my_input_shape)
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu"),
              input_shape=my_input_shape)
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    # flatten the output and feet it into dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

if __name__ == "__main__":
    # create train, validation, and test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(0.25, 0.2)
    
    # build the CNN
    my_input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (num_samples, 130, 13, 1)
    model = build_model(my_input_shape)

    # compile network
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=my_optimizer, loss="sparse_categorical_crossentropy", metrics =["accuracy"])
    model.fit(X_train, Y_train, 
        validation_data=(X_validation, Y_validation),
        batch_size=32,
        epochs=30)
    # train CNN
    test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    
    # eval CNN
    
    # make predictions 
    pass