import numpy as np


class neuron:
    def __init__(self, input_length, weights=None, bias = None):
        if weights is None:
            self.weights = np.random.rand(input_length)
        else:
            self.weights = weights
        if bias is None:
            self.bias = 0
        else:
            self.bias = bias

    def sumproduct(self, input):
        return np.sum(np.multiply(input, self.weights))


    def activation(self, sum):

        if sum + self.bias >= 0.5:
            return 1
        else:
            return 0


input = [0,1]
hidden_1 = neuron(2, [1., 1.], -0.5)
hidden_2 = neuron(2, [1., 1.], -1.5)
hidden_3 = neuron(2, [1., -2.], -0.5)

out_hidden1 = hidden_1.activation(hidden_1.sumproduct(input))
out_hidden2 = hidden_2.activation(hidden_2.sumproduct(input))
out_hidden3 = hidden_3.activation(hidden_3.sumproduct([out_hidden1, out_hidden2]))

print(out_hidden1, out_hidden2, out_hidden3)