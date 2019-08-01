import numpy as np
from tqdm import tqdm
from activ_funcs import *
from cost_funcs import *
from LearningRate import LRScheduler
from layers import *

# test with XOR gate
feature_set = np.array([[0, 0],[1, 1], [1,0], [0,1]])
labels = np.array([[0],[0],[1],[1]])

class Model:
    def __init__(self):
        self.layers = np.array([])

    def add_layer(self, layer):
        self.layers = np.append(self.layers, [layer])

    def save_model(self, name="model"):
        with open(name + ".arc", "w") as file:
            for x in self.layers:
                file.write(x.to_string() + " ")
            print("Model architecture saved to \'" + name + ".arc\'")

        with open(name + ".weights", "w") as file:
            for x in self.weights:
                file.write(np.array_repr(x) + "|")
            print("Model weights saved to \'" + name + ".weights\'")

    def load_architecture(self, name):
        with open(name, "r") as file:
            print("Loaded architecture...")
            self.layers = np.array([])
            arr = file.read()[:-1].split()
            
            for a in arr:
                info_arr = a.split("|")
                if info_arr[0] == "d":
                    self.add_layer(Dense(int(info_arr[1]), activation=info_arr[2]))
                else:
                    self.add_layer(Dense(int(info_arr[1]), name=info_arr[0], activation=info_arr[2]))
            print("Successfully loaded!")

    def load_weights(self, name):
        with open(name, "r") as file:
            print("Loaded weights...")
            self.weights = []
            arr = file.read()[:-1].split("|")
            
            for a in arr:
                self.weights.append(eval("np." + a))
            print("Successfully loaded!")

    def train(self, feature_set, labels, epochs=1000, momentum=0, **kwargs):
        # randomize starting weights and biases
        self.weights = []

        self.momentum = momentum
                
        # raise errors if not expected input shape
        if (feature_set.shape[-1] != self.layers[0].nodes):
            raise ValueError('Mismatching input dimentions (expected ' + str(self.layers[0].nodes) + ' but recieved ' +  str(feature_set.shape[-1]) + ')')
        elif (labels.shape[-1] != self.layers[-1].nodes):
            raise ValueError('Mismatching output dimentions (expected ' + str(self.layers[-1].nodes) + ' but recieved ' +  str(labels.shape[-1]) + ')')

        print(feature_set.shape[-1] )
        
        for i in range(1,self.layers.size):
            temp_arr = np.random.rand(self.layers[i-1].nodes,self.layers[i].nodes)
            self.weights.append(temp_arr)

        self.bias = np.random.rand(self.layers.size-1,1)

        # initializing learning rate scheduler
        LRS = LRScheduler()
        if ("learning_rate" in kwargs):
            LRS.constant(kwargs["learning_rate"])
        else:
            LRS.constant(0.5)

        retained_gradient = [np.zeros(i.shape) for i in self.weights]
        
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
                    dO_dnet = self.layers[i].d_activation(valMat[i+1])
                    
                    node_delta = dE_dO * dO_dnet
                    
                    
                    nodeDeltaMatrix.insert(i,node_delta)
                    #print(node_delta)
                    
                    dnet_dw = valMat[i]
                    #print(dnet_dw)
                    slope = np.matmul(dnet_dw.T, node_delta)


                    # update weights and bias
                    newWeights[i] -= (learning_rate * slope) + retained_gradient[i]
                    retained_gradient[i] = momentum*retained_gradient[i] + (learning_rate * slope)
                    self.bias[i] -= learning_rate * np.mean(slope)

                elif self.layers[i+1].name == "d":
                    dE_dO = np.matmul(self.weights[i+1],nodeDeltaMatrix[i+1][:].T).T

                    dO_dnet = self.layers[i].d_activation(valMat[i+1])
                    node_delta = dE_dO * dO_dnet

                    nodeDeltaMatrix.insert(i,node_delta)
                    
                    dnet_dw = valMat[i]

                    slope = np.matmul(dnet_dw.T, node_delta)
                    

                    # update weights and bias
                    newWeights[i] -= (learning_rate * slope) + retained_gradient[i]
                    retained_gradient[i] = momentum*retained_gradient[i] + (learning_rate * slope)
                    self.bias[i] -= learning_rate * np.mean(slope)

            # update weights for next pass
            self.weights = newWeights

    def predict(self,arr):
        inputs = arr
        for i in range(len(self.weights)):
            # calculate dot prod and apply activation func
            inputs = self.layers[i].activation(np.dot(inputs, self.weights[i]) + self.bias[i])
        return inputs
