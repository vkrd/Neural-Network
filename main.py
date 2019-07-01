import numpy as np
from tqdm import tqdm
from activ_funcs import *
from cost_funcs import *


layerType = np.array(["i", "d", "o"])

feature_set = np.array([[0.05, 0.1]])
labels = np.array([[0.01, 0.99]])
labels = labels.reshape(1, labels.shape[1])

np.random.seed(43)
learning_rate = 0.5

# define network architecture
arc = np.array([2, 2, 2])

# randomize starting weights and biases
weights = []

for nodes in range(1,arc.size):
    temp_arr = np.random.rand(arc[nodes-1],arc[nodes])
    weights.append(temp_arr)

bias = np.random.rand(arc.size-1,1)


def predict(arr):
    inputs = arr
    valMat = []
    inpMat = []
    for i in range(len(weights)):
        # calculate dot prod and apply activation func
        inpMat.append(np.dot(inputs, weights[i]) + bias[i])
        inputs = sigmoid(np.dot(inputs, weights[i]) + bias[i])
        valMat.append(inputs.copy())

for e in tqdm(range(10)):
    inputs = feature_set

    valMat = [inputs]

    # foreward prop
    for i in range(len(weights)):
        # calculate dot prod and apply activation func
        inputs = sigmoid(np.dot(inputs, weights[i]) + bias[i])
        # print(inputs)
        valMat.append(inputs.copy())

    # output layer
    output_vec = inputs
    target = labels

    # backwards prop
    nodeDeltaMatrix = [np.empty_like(i) for i in weights]
    newWeights = [np.copy(i) for i in weights]

    for i in reversed(range(len(weights))):
        if layerType[i + 1] == "o":

            dE_dO = d_MSE(output_vec, target)
            dO_dnet = d_sigmoid(valMat[i+1])
            node_delta = dE_dO * dO_dnet
            nodeDeltaMatrix.insert(i,node_delta)

            dnet_dw = valMat[i]

            slope = np.matmul(dnet_dw.T, node_delta)

            newWeights[i] -= (learning_rate * slope)


            bias[i] -= learning_rate * np.mean(slope)

        elif layerType[i+1] == "d":
            dE_dO = np.matmul(weights[i+1],nodeDeltaMatrix[i+1][:].T).T

            dO_dnet = d_sigmoid(valMat[i+1])
            node_delta = dE_dO * dO_dnet

            dnet_dw = valMat[i]

            slope = np.matmul(dnet_dw.T, node_delta)

            newWeights[i] = weights[i] - (learning_rate * slope)

            bias[i] -= learning_rate * np.mean(slope)

    #update weights for next pass
    weights = newWeights