import numpy as np
from tqdm import tqdm
from cost_funcs import *
from LearningRate import LRScheduler
from layers import *

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

class Model:
    def __init__(self):
        self.layers = np.array([])

    def add_layer(self, layer):
        self.layers = np.append(self.layers, [layer])

    def train(self, feature_set, labels, epochs=1000, **kwargs):
        # randomize starting weights and biases
        self.weights = []

        # raise errors if not expected input shape
        if feature_set.shape[-1] != self.layers[0].nodes:
            raise ValueError(
                'Mismatching input dimensions (expected ' + str(self.layers[0].nodes) + ' but received ' + str(
                    feature_set.shape[-1]) + ')')
        elif labels.shape[-1] != self.layers[-1].nodes:
            raise ValueError(
                'Mismatching output dimensions (expected ' + str(self.layers[-1].nodes) + ' but received ' + str(
                    labels.shape[-1]) + ')')

        for i in range(1, self.layers.size):
            temp_arr = np.random.rand(self.layers[i - 1].nodes, self.layers[i].nodes)
            self.weights.append(temp_arr)

        self.bias = np.random.rand(self.layers.size - 1, 1)

        # initializing learning rate scheduler
        LRS = LRScheduler()
        if ("learning_rate" in kwargs):
            LRS.constant(kwargs["learning_rate"])
        else:
            LRS.constant(0.5)

        for e in tqdm(range(epochs)):
            inputs = feature_set

            valMat = [inputs]

            # foreward prop                
            for i in range(len(self.weights)):
                # calculate dot prod and apply activation func
                inputs = self.layers[i].activation(np.dot(inputs, self.weights[i]) + self.bias[i])
                valMat.append(inputs.copy())

            # output layer
            output_vec = inputs
            target = labels

            # backwards prop
            nodeDeltaMatrix = [np.empty_like(i) for i in self.weights]
            newWeights = [np.copy(i) for i in self.weights]

            learning_rate = LRS.nextLR()

            for i in reversed(range(len(self.weights))):
                if self.layers[i + 1].name == "o":

                    dE_dO = d_MSE(output_vec, target)
                    dO_dnet = self.layers[i].d_activation(valMat[i + 1])

                    node_delta = dE_dO * dO_dnet

                    nodeDeltaMatrix.insert(i, node_delta)
                    # print(node_delta)

                    dnet_dw = valMat[i]
                    # print(dnet_dw)
                    slope = np.matmul(dnet_dw.T, node_delta)

                    # update weights and bias
                    newWeights[i] -= (learning_rate * slope)
                    self.bias[i] -= learning_rate * np.mean(slope)

                elif self.layers[i + 1].name == "d":
                    dE_dO = np.matmul(self.weights[i + 1], nodeDeltaMatrix[i + 1][:].T).T

                    dO_dnet = self.layers[i].d_activation(valMat[i + 1])
                    node_delta = dE_dO * dO_dnet

                    nodeDeltaMatrix.insert(i, node_delta)

                    dnet_dw = valMat[i]

                    slope = np.matmul(dnet_dw.T, node_delta)

                    # update weights and bias
                    newWeights[i] -= (learning_rate * slope)
                    self.bias[i] -= learning_rate * np.mean(slope)

            # update weights for next pass
            self.weights = newWeights

    def predict(self, arr):
        inputs = arr
        for i in range(len(self.weights)):
            # calculate dot prod and apply activation func
            inputs = self.layers[i].activation(np.dot(inputs, self.weights[i]) + self.bias[i])
        return inputs
