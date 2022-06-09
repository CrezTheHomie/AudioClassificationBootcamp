import numpy as np

# save activations and derivatives

# calculate error
# implement training
# train our network with a dummy dataset

# multi layered perception
class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3,3], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [num_inputs] + num_hidden + [num_outputs]

        # we'll have random weights
        weights = []
        for i in range(len(layers)-1):
            w =  np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def _sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        for i,w in enumerate(self.weights):
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            # calculate activation
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        return activations

    # implement backpropagation
    def back_propagage(self, error, verbose=False):
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

            error = np.dot(delta, self.weights[i].T)
            if verbose:
                print("Derivatives for W{}: {} \n".format(i, self.derivatives))
        return error

    def gradient_descent(self, learning_rate):

        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("Original W{} {}".format(i, weights))

            derivatives = self.derivatives[i]
            weights += (derivatives * learning_rate)
            #print("Updated W{} {}".format(i, weights))
    
    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for j, (input, target) in enumerate(zip(inputs, targets)):

                #forward propagate
                output = self.forward_propagate(input, target)

                # calculate error
                error = target - output

                # backward propagate
                self.back_propagage(error)

                # apply gradient descent
                mlp.gradient_descent(learning_rate)
                
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

                # report error for each epoch
                print("Error is {} at epoch {}".format(sum_error/len(inputs), i))

    def _mse(self, target, output):
        return np.average((target-output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

if __name__ == "__main__":

    # create and MLP
    mlp = MLP(2, [5], 1)
    # create 
    #inputs = np.random.rand(mlp.num_inputs)
    inputs = np.array([0.1,0.2])
    target = np.array([0.3])
    # forward propagate
    outputs = mlp.forward_propagate(inputs)
    # print results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))