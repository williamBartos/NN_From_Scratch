import numpy as np


class neuralnet:
    def __init__(self, nodes_per_layer, activation_fn, cost_fn):
        self.num_layers = len(nodes_per_layer)
        self.num_nodes_per_layer = nodes_per_layer
        self.activation_fn = activation_fn
        self.cost_fn = cost_fn
        self.layers = []

        for i in range(self.num_layers):
            # i = input dim, i+1 = output dim
            if i == self.num_layers - 1:
                layer_i = layer(nodes_per_layer[i], 0, activation_fn[i])
            else:
                layer_i = layer(nodes_per_layer[i], nodes_per_layer[i+1], activation_fn[i])

            self.layers.append(layer_i)

    def forward_pass(self, inputs):
        self.layers[0].activations = inputs
        for i in range(self.num_layers-1):
            input_products = np.add(np.matmul(self.layers[i].activations, self.layers[i].weights), self.layers[i].bias)
            if self.activation_fn == "sigmoid":
                self.layers[i+1].activations=self.sigmoid(input_products)
            elif self.activation_fn == "relu":
                self.layers[i+1].activations=self.relu(input_products)
            else:
                self.layers[i+1].activations = input_products


    def sigmoid(self, layer):
        return np.divide(1, np.exp(np.negative(layer)))

    def relu(self, layer):
        for i in layer:
            layer[i] = max(0, layer[i])
        return layer

class layer:
    def __init__(self, nodes_input, nodes_output, activation_fn):
        self.nodes_input = nodes_input
        self.nodes_output = nodes_output
        self.activation_fn = activation_fn
        self.activations = np.zeros([nodes_input,1])

        if nodes_output != 0:
            self.weights = np.random.normal(0, 0.001, size=(nodes_input, nodes_output))
            self.bias = np.random.normal(0, 0.001, size=(1, nodes_output))
        else:
            self.weights = None
            self.bias = None

inputs = [0]
network = neuralnet([1,2,1], [None, None, None], cost_fn="cross_entropy")
network.forward_pass(inputs)
print(network.layers[2].activations)