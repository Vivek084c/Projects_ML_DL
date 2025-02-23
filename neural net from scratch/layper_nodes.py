import numpy as np


np.random.seed(0)

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

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

layer1 = Layers_Dense(4, 5)
layer2 = Layers_Dense(5, 2)

layer1.forward(X)
print(layer1.output)

#passing the output from layer 1 --> layer 2
layer2.forward(layer1.output)
print(layer2.output)


# print(np.zeros((1,3)).astype(np.dtype('int16')))