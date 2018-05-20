import numpy as np


class neuralnet:
    def __init__(self, nodes_per_layer, activation_fn, cost_fn):
        self.num_layers = len(nodes_per_layer)
        self.num_nodes_per_layer = nodes_per_layer
        self.activation_fn = activation_fn
        self.cost_fn = cost_fn
        self.layers = []
        self.error = 0

        for i in range(self.num_layers):
            # i: input dim, i+1: output dim
            if i == self.num_layers - 1:
                layer_i = layer(nodes_per_layer[i], 0, activation_fn[i])
            else:
                layer_i = layer(nodes_per_layer[i], nodes_per_layer[i+1], activation_fn[i])

            self.layers.append(layer_i)

    def forward_pass(self, inputs):
        self.layers[0].activations = inputs
        for i in range(self.num_layers-1):
            # input_products are the elementwise matrix product and summation of the previous node outputs and weights
            # We iterate to the i-1th layer because the final/output layer is just the activations of the last hidden layer
            # Passed through the putput layer's activation function
            input_products = np.add(np.matmul(self.layers[i].activations, self.layers[i].weights), self.layers[i].bias)
            if self.layers[i+1].activation_fn == "sigmoid":
                self.layers[i+1].activations=self.sigmoid(input_products)
            elif self.layers[i+1].avtivation_fn == "relu":
                self.layers[i+1].activations=self.relu(input_products)
            else:
                self.layers[i+1].activations = input_products

    def sigmoid(self, layer):
        return np.divide(1, np.exp(np.negative(layer)))

    def relu(self, layer):
        for i in layer:
            layer[i] = max(0, layer[i])
        return layer

    def loss_fn(self,labels):
        if len(labels[0]) != len(self.layers[self.num_layers - 1].nodes_output):
            print("Error, wrong label shape")
            return
        self.error = self.cross_entropy(labels)

    def cross_entropy(self, labels):
        out_layer_activations = self.layers[self.num_layers-1].activations
        self.error += -np.sum(np.multiply(labels, np.log(out_layer_activations)) + np.multiply((np.subtract(1, labels)), np.log(out_layer_activations)))





class layer:
    def __init__(self, nodes_input, nodes_output, activation_fn):
        self.nodes_input = nodes_input
        self.nodes_output = nodes_output
        self.activation_fn = activation_fn
        self.activations = np.zeros([nodes_input,1])

        if nodes_output != 0:
            # Weight matrix dimensions are a combination of the input and output nodes
            # Bias is just applied to the outputting nodes
            self.weights = np.random.normal(0, 0.001, size=(nodes_input, nodes_output))
            self.bias = np.random.normal(0, 0.001, size=(1, nodes_output))
        else:
            self.weights = None
            self.bias = None

inputs = [1,2]
network = neuralnet([2,2,1], [None, "sigmoid", "sigmoid"], cost_fn="cross_entropy")
network.forward_pass(inputs)
network.cross_entropy([1])
network.cross_entropy([0])

print(network.error)

