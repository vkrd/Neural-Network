import numpy as np
from activ_funcs import *
from cost_funcs import *

feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])  
labels = np.array([[1,0,0,1,1]])  
labels = labels.reshape(5,1)

np.random.seed(69)
weights = np.random.rand(3,1)
bias = np.random.rand(1)

learning_rate = 0.03

for e in range(10000):
    inputs = feature_set

    #calculate dot product
    output_vec = np.dot(inputs, weights)+bias

    #apply activation function
    output_vec = sigmoid(output_vec)

    #calculate gradient
    error = output_vec - labels
    dpred_dcost = d_sigmoid(output_vec)
    slope = np.dot(inputs.T, error*dpred_dcost) 

    #update weights and bias
    weights -= learning_rate * slope

    for num in error*dpred_dcost:
        bias -= learning_rate * num

    print(MSE(output_vec,labels))
