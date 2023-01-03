# import what we need and set up globals

import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import _keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "genres_original"
JSON_PATH = "data\GTZAN_dict.json"

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def prepare_datasets(my_test_size, validation_size):
    # load the data
    X, y = load_data(JSON_PATH)

    # create the train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=my_test_size)

    # create the validation test split
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(my_input_shape):

    # create model
    model = keras.Sequential()

    # 2 lstm layers: seq to seq and seq to vector
    model.add(keras.layers.LSTM(64, input_shape=my_input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64)) 

    # flatten the output and feet it into dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

def predict(model, X, y):
    X = X[np.newaxis, ...]

    predictions = model.predict(X)
    print("\nPrediction index is {}".format(predictions))

    mfcc, labels = load_data(JSON_PATH)
    predicted_index = np.argmax(predictions, axis=1)
    print("Expected index: {}, Predicted index: {}".format(
        y, predicted_index))

    return predictions


if __name__ == "__main__":
    # create train, validation, and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2)

    # (130, 13)
    my_input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(my_input_shape)

    # compile network
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=my_optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # train CNN
    history = model.fit(X_train, y_train,
              validation_data=(X_validation, y_validation),
              batch_size=32,
              epochs=30)

    plot_history(history)

    # eval CNN
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make predictions
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)
