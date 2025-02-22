import numpy as np

input1 = [1.5, 2.5, 1, 1]
weights1 = [0.2, 0.8, -0.5, 1]
input2 = [1.5, 2.5, 1, 1]
weights2 = [0.2, 0.8, -0.5, 1]
input3 = [1.5, 2.5, 1, 1]
weights3 = [0.2, 0.8, -0.5, 1]

bias1= 1
bias2= 2
bias3= 3
 
otuput = [0, 0, 0]
otuput[0] = input1[0]*weights1[0] + input1[1]*weights1[1] + input1[2]*weights1[2] + input1[3]*weights1[3] + bias1
otuput[1] = input2[0]*weights2[0] + input2[2]*weights2[1] + input2[2]*weights2[2] + input2[3]*weights2[3] + bias2
otuput[2] = input3[0]*weights3[0] + input3[1]*weights3[1] + input3[2]*weights3[2] + input3[3]*weights3[3] + bias3

print(otuput[0])
print(otuput[1])
print(otuput[2])

