import math

# Code along with Valerio Velardo - The Sound of AI

# activation function of choice [can be anything]
def sigmoid(x):
    y = 1.0 / (1+ math.exp(-x))
    return y

# activation function
def activate(inputs,weights):
    # perform net input
    h = 0
    for x, w in zip(inputs,weights):
        h += x * w
    # perform the activation
    return sigmoid(h)

# test the neuron
if __name__ == "__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output = activate(inputs,weights)
    print(output)
