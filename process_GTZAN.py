from preprocess_GTZAN import *
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

def load_data(DATASET_PATH):
    with open(JSON_PATH, "r") as fp:
        data = json.load(fp)

    # convert lists into np arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")

    # create error/loss subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("loss Eval")

    plt.show()
    

if __name__ == "__main__":
    # load data
    inputs, targets = load_data(DATASET_PATH)
    # split data 
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3, random_state=0)

    # build ML architecture
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        # 1st hidden layer
        keras.layers.Dense(512, activation="relu",
                           kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.Dropout(0.4),
        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu",
                           kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.Dropout(0.3),
        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.Dropout(0.1),
        # output layer with the 10 categories of our data's mapping
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile
    my_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer = my_optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # train
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs = 50, batch_size=32)
    
    # from our training set, our output is looking good, but our output for our validation set, our model performs poorly
    # we have overfitted! Are we going to dropout, regularization, early stop or other?

    # lets plot the accuracy and error over the epochs
    plot_history(history)