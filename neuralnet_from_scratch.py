import numpy as np

# save activations and derivatives
# implement backpropagation
# calculate error
# implement training
# train our network with a dummy dataset

# multi layered perception
class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [num_inputs] + num_hidden + [num_outputs]

        # we'll have random weights
        self.random_weights = []
        for i in range(len(layers)-1):
            w =  np.random.rand(layers[i], layers[i+1])
            random_weights.append(w)
        self.random_weights = random_weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
        derivatives = []
        for i in range(len(layers)):
            d = np.zeros(layers[i], layers[i+1])
            derivatives.append(d)
        self.derivatives = derivatives

    def _sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        for w in self.random_weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            # calculate activation
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        return activations

    def back_propagage(self, error):
        # loop through all the layers backwards

        # dE/dW_i = ( y - a[i+1]) (s'(h_[i+1])) a_i 
        # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        # s_(h[i+1]) = a_[i+1] 

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

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