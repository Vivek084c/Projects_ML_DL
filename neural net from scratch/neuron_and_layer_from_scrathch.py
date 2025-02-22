import numpy as np

inputs= [1, 2, 3, 2.5]
weights =[
    [0.2, 0.8, -0.5, 1],
    [0.2, 0.8, -0.5, 1],
    [0.2, 0.8, -0.5, 1]
]

bias = [1, 2, 3]
 
layer_output = []
for neuron_weight, neuron_bias in zip(weights, bias):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weight):
        neuron_output += n_input * weight
    neuron_output += neuron_bias

    layer_output.append(neuron_output)

print(layer_output)




