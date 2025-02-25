import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()







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

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss 
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)  

        if len(y_true.shape) == 1:
            correct_confidences = y_pred[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred*y_true, axis=1) 

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(samples = 100, classes= 3)

# defining the object for dense network class, acitvation relu class and acitvatation softmax class
dense1 = Layers_Dense(2, 3)
activation_1 = Activation_ReLU()

dense2 = Layers_Dense(3, 3) 
activation_2 = Activation_Softmax()


dense1.forward(X)
activation_1.forward(dense1.output)

dense2.forward(activation_1.output)
activation_2.forward(dense2.output)

print(activation_2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss_function.calculate(activation_2.output, y)


