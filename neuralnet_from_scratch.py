import numpy as np

# multi layered perception
class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # we'll have random weights
        self.random_weights = []
        for i in range(len(layers)-1):
            w =  np.random.rand(layers[i], layers[i+1])
            self.random_weights.append(w)

    def _sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        activations = inputs

        for w in self.random_weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            # calculate activation
            activations = self._sigmoid(net_inputs)
        return activations

if __name__ == "__main__":

    # create and MLP
    mlp = MLP()
    # create 
    inputs = np.random.rand(mlp.num_inputs)
    # perform
    outputs = mlp.forward_propagate(inputs)
    # print results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))