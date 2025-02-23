import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

X, y = spiral_data(100, 3)


print(X[0].shape)
# inputs = [0, 2, -1, 3.3, 2.7, 1.1, 2.2, -100]
# output = []

# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

# print(output)


class Layers_Dense:
    def __init__(self, NoOfInputs, NoOfNeurons):
        """
        NoOfInputs : no of inputs to this neuron
        NoOfNeurons : no of neurons in the layer
        """
        self.weights = np.random.randn(NoOfInputs, NoOfNeurons) 
        """defined as noofinputs x noofneurons 
        to prevent the use of tranoose"""
        self.bias = np.zeros((1, NoOfNeurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


#defining a layer with 2 inputs
layer1 = Layers_Dense(2, 5)
#defining the relu acitvation 
activatation_1 = Activation_ReLU()

layer1.forward(X)
activatation_1.forward(layer1.output)

print(layer1.output)
print(activatation_1.output)

