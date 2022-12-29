from preprocess_GTZAN import *
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

def load_data(DATASET_PATH):
    with open(JSON_PATH, "r") as fp:
        data = json.load(fp)

    # convert lists into np arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


# split data between test and train
# build ML arch



if __name__ == "__main__":
    # load data
    inputs, targets = load_data(DATASET_PATH)
    # split data 
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3, random_state=0)

    # build ML architecture
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        # 1st hidden layer
        keras.layers.Dense(512, activation="relu"),
        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu"),
        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu"),
        # output layer with the 10 categories of our data's mapping
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer = my_optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # train
    model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs = 50, batch_size=32)
    
    # from our training set, our output is looking good, but our output for our validation set, our model performs poorly
    # we have overfitted!